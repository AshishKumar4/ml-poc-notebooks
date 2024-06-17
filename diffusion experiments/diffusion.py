"""
This is just a simple implementation of diffusion techniques by Ashish Kumar Singh
based on understanding from keras tutorials and other implementations as well
as the paper
"""

# %%
%load_ext dotenv
%dotenv

import tqdm
import flax
from flax import linen as nn
import jax
from typing import Dict, Callable, Sequence, Any, Union
from dataclasses import field
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import einops
import clu
from clu import parameter_overview, metrics
from flax.training import train_state  # Useful dataclass to keep train state
import optax
from flax import struct                # Flax dataclasses
import time
import os
from datetime import datetime
import orbax
from flax.training import orbax_utils
import numpy as np


# %%
normalizeImage = lambda x: jax.nn.standardize(x, mean=[127.5], std=[127.5])
denormalizeImage = lambda x: (x + 1.0) * 127.5


def plotImages(imgs, fig_size=(8, 8), dpi=100):
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    imglen = imgs.shape[0]
    for i in range(imglen):
        plt.subplot(fig_size[0], fig_size[1], i + 1)
        plt.imshow(tf.cast(denormalizeImage(imgs[i, :, :, :]), tf.uint8))
        plt.axis("off")
    plt.show()

class RandomClass():
    def __init__(self, rng: jax.random.PRNGKey):
        self.rng = rng

    def get_random_key(self):
        self.rng, subkey = jax.random.split(self.rng)
        return subkey
    
    def set_random_key(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def reset_random_key(self):
        self.rng = jax.random.PRNGKey(42)

class MarkovState(struct.PyTreeNode):
    pass

class RandomMarkovState(MarkovState):
    rng: jax.random.PRNGKey

    def get_random_key(self):
        rng, subkey = jax.random.split(self.rng)
        return RandomMarkovState(rng), subkey

# %% [markdown]
# # Data Pipeline

# %%
def get_dataset(data_name="celeb_a", batch_size=64, image_scale=256):
    def augmenter(image_scale=256, method="area"):
        @tf.function()
        def augment(sample):
            image = (
                tf.cast(sample["image"], tf.float32) - 127.5
            ) / 127.5
            image = tf.image.resize(
                image, [image_scale, image_scale], method=method, antialias=True
            )
            image = tf.image.random_flip_left_right(image)
            return image
        return augment

    # Load CelebA Dataset
    data: tf.data.Dataset = tfds.load(data_name, split="all", shuffle_files=True)
    final_data = (
        data
        # .prefetch(tf.data.experimental.AUTOTUNE)
        # .batch(128, num_parallel_calls=tf.data.AUTOTUNE)
        .map(
            augmenter(image_scale, method="area"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .cache()  # Cache after augmenting to avoid recomputation
        .repeat()  # Repeats the dataset indefinitely
        # .unbatch()
        .shuffle(4096)  # Ensure this is adequate for your dataset size
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    ).as_numpy_iterator()
    return final_data, len(data)


# %% [markdown]
# # Noise Samplers

# %%
def cosine_beta_schedule(timesteps, start_angle=0.008, end_angle=0.999):
    ts = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
    alphas_bar = np.cos((ts + start_angle) / (1 + start_angle) * np.pi /2) ** 2
    alphas_bar = alphas_bar/alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, end_angle)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    betas = np.linspace(
        beta_start, beta_end, timesteps, dtype=np.float64)
    return betas

class NoiseScheduler():
    def __init__(self, timesteps,
                    dtype=jnp.float32,
                    clip_min=-1.0,
                    clip_max=1.0,
                    *args, **kwargs):
        self.max_timesteps = timesteps
        self.dtype = dtype
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        raise NotImplementedError
    
    def get_p2_weights(self, k, gamma):
        raise NotImplementedError
    
    def clip_images(self, images) -> jnp.ndarray:
        raise NotImplementedError
    
    def reshape_rates(self, rates:tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
    
    def get_rates(self, steps):
        raise NotImplementedError
    
    def add_noise(self, images, noise, steps, rates=None) -> jnp.ndarray:
        raise NotImplementedError
    
    def remove_all_noise(self, noisy_images, noise, steps, clip_denoised=True, rates=None):
        raise NotImplementedError
    
    def transform_steps(self, steps):
        raise NotImplementedError

class DiscreteNoiseScheduler(NoiseScheduler):
    def __init__(self, timesteps,
                    beta_start=0.0001,
                    beta_end=0.02,
                    schedule_fn=linear_beta_schedule, 
                    *args, **kwargs):
        super().__init__(timesteps, *args, **kwargs)
        betas = schedule_fn(timesteps, beta_start, beta_end)
        alphas = 1 - betas
        alpha_cumprod = jnp.cumprod(alphas, axis=0)
        alpha_cumprod_prev = jnp.append(1.0, alpha_cumprod[:-1])
        
        self.betas = jnp.array(betas, dtype=jnp.float32)
        self.alphas = alphas.astype(jnp.float32)
        self.alpha_cumprod = alpha_cumprod.astype(jnp.float32)
        self.alpha_cumprod_prev = alpha_cumprod_prev.astype(jnp.float32)

        self.sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod).astype(jnp.float32)
        self.sqrt_one_minus_alpha_cumprod = jnp.sqrt(1 - alpha_cumprod).astype(jnp.float32)

        self.sqrt_recip_alpha_cumprod = jnp.sqrt(1 / alpha_cumprod).astype(jnp.float32)
        self.sqrt_recip_one_minus_alpha_cumprod = jnp.sqrt(1 / alpha_cumprod - 1).astype(jnp.float32)

        posterior_variance = (betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
        self.posterior_variance = posterior_variance.astype(jnp.float32)
        self.posterior_log_variance_clipped = (jnp.log(jnp.maximum(posterior_variance, 1e-20))).astype(jnp.float32)
        
        self.posterior_mean_coef1 = (betas * jnp.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod)).astype(jnp.float32)
        self.posterior_mean_coef2 = ((1 - alpha_cumprod_prev) * jnp.sqrt(alphas) / (1 - alpha_cumprod)).astype(jnp.float32)

    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        timesteps = jax.random.randint(rng, (batch_size,), 0, self.max_timesteps)
        return timesteps, state
    
    def get_p2_weights(self, k, gamma):
        return (k + self.alpha_cumprod / (1 - self.alpha_cumprod)) ** -gamma
    
    def clip_images(self, images) -> jnp.ndarray:
        return jnp.clip(images, self.clip_min, self.clip_max)
    
    def reshape_rates(self, rates:tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        signal_rates, noise_rates = rates
        signal_rates = jnp.reshape(signal_rates, (-1, 1, 1, 1))
        noise_rates = jnp.reshape(noise_rates, (-1, 1, 1, 1))
        return signal_rates, noise_rates

    def get_rates(self, steps):
        signal_rate = self.sqrt_alpha_cumprod[steps]
        noise_rate = self.sqrt_one_minus_alpha_cumprod[steps]
        return signal_rate, noise_rate
    
    # Used while training
    def add_noise(self, images, noise, steps) -> jnp.ndarray:
        rates = self.get_rates(steps)
        signal_rates, noise_rates = self.reshape_rates(rates)
        return signal_rates * images + noise_rates * noise

    def remove_all_noise(self, noisy_images, noise, steps):
        # Scale 't' to the range [0, 1]
        signal_coeff = self.sqrt_recip_alpha_cumprod[steps]
        noise_coeff = self.sqrt_recip_one_minus_alpha_cumprod[steps]
        signal_coeff, noise_coeff = self.reshape_rates((signal_coeff, noise_coeff))
        pred_images = signal_coeff * noisy_images - noise_coeff * noise
        # pred_images = (noisy_images - noise_rate * noise) / signal_rate
        pred_images = jnp.clip(pred_images, -1, 1)
        return pred_images
    
    # Used while training
    def transform_steps(self, steps):
        return steps #/ self.max_timesteps

class CosineNoiseSchedule(DiscreteNoiseScheduler):
    def __init__(self, timesteps, beta_start=0.008, beta_end=0.999, *args, **kwargs):
        super().__init__(timesteps, beta_start, beta_end, schedule_fn=cosine_beta_schedule, *args, **kwargs)

class LinearNoiseSchedule(DiscreteNoiseScheduler):
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, *args, **kwargs):
        super().__init__(timesteps, beta_start, beta_end, schedule_fn=linear_beta_schedule, *args, **kwargs)

# %%
schedule = CosineNoiseSchedule(1000)
plt.plot(schedule.betas)
# plt.plot(sampler.alphas)
# plt.plot(sampler.alpha_cumprod)
# plt.plot(sampler.alpha_cumprod_prev)
plt.plot([schedule.get_rates(i) for i in range(1000)])
deltas = []
for i in range(1000):
    signal, noise = schedule.get_rates(i)
    deltas.append(jnp.abs(1 - (signal**2 + noise**2)))
print(sum(deltas))

# %% [markdown]
# # Data Diffusion Testing

# %%
# Visualize adding some noise to some sample images and then removing it
state = RandomMarkovState(jax.random.PRNGKey(4))
data, _ = get_dataset("oxford_flowers102", batch_size=4, image_scale=256)
images = next(iter(data))
plotImages(images)
noise_level = 400# * jnp.ones((images.shape[0], ), dtype=jnp.int32)
noise_level_max = 1000
schedule = CosineNoiseSchedule(noise_level_max)
sampler = DDPMSampler(schedule)
state, rng = state.get_random_key()
noise = jax.random.normal(rng, shape=images.shape, dtype=jnp.float64)
noisy_images = schedule.add_noise(images, noise, noise_level)
plotImages(noisy_images)
plotImages(noise)
reconstructed_images = schedule.remove_all_noise(noisy_images, noise, noise_level)
plotImages(reconstructed_images)

# Now test sampling
# sample = jax.random.normal(random.get_random_key(), shape=images.shape, dtype=jnp.float64)
sample = noisy_images
for i in reversed(range(0, noise_level)):
    step = jnp.ones((images.shape[0], 1, 1, 1), dtype=jnp.int32) * i
    sample, state = sampler.sample_step(sample, noise, step, state=state)
        # sample = schedule.remove_all_noise(sample, noise, step, clip_denoised=False)

plotImages(sample)
print(schedule.get_rates(noise_level))

# %% [markdown]
# # Modeling

# %% [markdown]
# ## Metrics

# %% [markdown]
# ## Callbacks

# %% [markdown]
# ## Model Generator

# %%
# Built using the Flax NNX library

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return nn.initializers.variance_scaling(scale=scale, mode="fan_avg", distribution="normal")

class WeightStandardizedConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """ 
    features: int
    kernel_size: Sequence[int] = 3
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32


    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)
        
        conv = nn.Conv(
            features=self.features, 
            kernel_size=self.kernel_size, 
            strides = self.strides,
            padding=self.padding, 
            dtype=self.dtype, 
            param_dtype = self.param_dtype,
            parent=None)
        
        # standardize kernel
        kernel = self.param('kernel', kernel_init(1.0), x)
        eps = 1e-5 if self.dtype == jnp.float32 else 1e-3
        # reduce over dim_out
        redux = tuple(range(kernel.ndim - 1))
        mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = (kernel - mean)/jnp.sqrt(var + eps)

        bias = self.param('bias',kernel_init(1.0), x)

        return(conv.apply({'params': {'kernel': standardized_kernel, 'bias': bias}},x))
    
class PixelShuffle(nn.Module):
    scale: int

    @nn.compact
    def __call__(self, x):
        up = einops.rearrange(
            x,
            pattern="b h w (h2 w2 c) -> b (h h2) (w w2) c",
            h2=self.scale,
            w2=self.scale,
        )
        return up
    
class TimeEmbedding(nn.Module):
    features:int
    max_timesteps:int=10000

    def setup(self):
        # self.embeddings = nn.Embed(
        #     num_embeddings=max_timesteps, features=out_features
        # )
        half_dim = self.features // 2
        emb = jnp.log(self.max_timesteps) / (half_dim - 1)
        emb = jnp.exp(-emb * jnp.arange(half_dim, dtype=jnp.float32))
        self.embeddings = emb
    
    def __call__(self, x):
        x = jax.lax.convert_element_type(x, jnp.float32)
        emb = x[:, None] * self.embeddings[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb
    
class TimeProjection(nn.Module):
    features:int
    activation:Callable=jax.nn.gelu

    @nn.compact
    def __call__(self, x):
        x = nn.DenseGeneral(self.features, kernel_init=kernel_init(1.0))(x)
        x = self.activation(x)
        x = nn.DenseGeneral(self.features, kernel_init=kernel_init(1.0))(x)
        x = self.activation(x)
        return x
    
class SeparableConv(nn.Module):
    features:int
    kernel_size:tuple=(3, 3)
    strides:tuple=(1, 1)
    use_bias:bool=False
    kernel_init:Callable=kernel_init(1.0)
    padding:str="SAME"
    
    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        depthwise = nn.Conv(
            features=in_features, kernel_size=self.kernel_size,
            strides=self.strides, kernel_init=self.kernel_init,
            feature_group_count=in_features, use_bias=self.use_bias,
            padding=self.padding
        )(x)
        pointwise = nn.Conv(
            features=self.features, kernel_size=(1, 1),
            strides=(1, 1), kernel_init=self.kernel_init,
            use_bias=self.use_bias
        )(depthwise)
        return pointwise


class ConvLayer(nn.Module):
    conv_type:str
    features:int
    kernel_size:tuple=(3, 3)
    strides:tuple=(1, 1)
    kernel_init:Callable=kernel_init(1.0)

    def setup(self):
        # conv_type can be "conv", "separable", "conv_transpose"
        if self.conv_type == "conv":
            self.conv = nn.Conv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                kernel_init=self.kernel_init,
            )
        elif self.conv_type == "w_conv":
            self.conv = WeightStandardizedConv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="SAME",
                dtype=jnp.float32,
                param_dtype=jnp.float32
            )
        elif self.conv_type == "separable":
            self.conv = SeparableConv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                kernel_init=self.kernel_init,
            )
        elif self.conv_type == "conv_transpose":
            self.conv = nn.ConvTranspose(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                kernel_init=self.kernel_init,
            )

    def __call__(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    features:int
    scale:int
    activation:Callable=jax.nn.mish

    @nn.compact
    def __call__(self, x, residual=None):
        out = x
        out = PixelShuffle(scale=self.scale)(out)
        # B, H, W, C = x.shape
        # out = jax.image.resize(x, (B, H * self.scale, W * self.scale, C), method="nearest")
        out = ConvLayer(
            "conv",
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
        )(out)
        if residual is not None:
            out = jnp.concatenate([out, residual], axis=-1)
        return out
    
class Downsample(nn.Module):
    features:int
    scale:int
    activation:Callable=jax.nn.mish
    
    @nn.compact
    def __call__(self, x, residual=None):
        out = ConvLayer(
            "conv",
            features=self.features,
            kernel_size=(3, 3),
            strides=(2, 2)
        )(x)
        if residual is not None:
            if residual.shape[1] > out.shape[1]:
                residual = nn.avg_pool(residual, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            out = jnp.concatenate([out, residual], axis=-1)
        return out
    
class CrossAttention(nn.Module):
    num_heads: int
    head_dim: int
    kernel_init: Callable = kernel_init(1.0)

    @nn.compact
    def __call__(self, x, context):
        q = nn.Dense(self.num_heads * self.head_dim, kernel_init=self.kernel_init)(x)
        k = nn.Dense(self.num_heads * self.head_dim, kernel_init=self.kernel_init)(context)
        v = nn.Dense(self.num_heads * self.head_dim, kernel_init=self.kernel_init)(context)

        q = q.reshape(x.shape[0], -1, self.num_heads, self.head_dim)
        k = k.reshape(context.shape[0], -1, self.num_heads, self.head_dim)
        v = v.reshape(context.shape[0], -1, self.num_heads, self.head_dim)

        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_output = attn_output.reshape(x.shape[0], -1, self.num_heads * self.head_dim)

        return nn.Dense(x.shape[-1])(attn_output)
    
class AttentionBlock(nn.Module):
    attention_config:Dict[str, Any]
    kernel_init:Callable=kernel_init(1.0)
    attention_kwargs:Dict[str, Any]=field(default_factory=dict)

    # def setup(self) -> None:
    #     if self.attention_config.get('attention_fn', None) != None:
    #         self.attention_fn = self.attention_config['attention_fn']
    #     return 

    @nn.compact
    def __call__(self, x):
        attention = nn.MultiHeadAttention(**self.attention_config, kernel_init=self.kernel_init, decode=False)(x)
        out = x + attention
        return out

class ResidualBlock(nn.Module):
    conv_type:str
    features:int
    kernel_size:tuple=(3, 3)
    strides:tuple=(1, 1)
    padding:str="SAME"
    activation:Callable=jax.nn.mish
    direction:str=None
    res:int=2
    norm_groups:int=8
    kernel_init:Callable=kernel_init(1.0)

    @nn.compact
    def __call__(self, x:jax.Array, temb:jax.Array, extra_features:jax.Array=None):
        residual = x
        out = nn.GroupNorm(8)(x)
        out = self.activation(out)

        out = ConvLayer(
            self.conv_type, 
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=self.kernel_init,
        )(out)
        out = nn.GroupNorm(8)(out)

        temb = nn.DenseGeneral(features=self.features*2)(temb)[:, None, None, :]
        scale, shift = jnp.split(temb, 2, axis=-1)
        out = out * (1 + scale) + shift

        out = self.activation(out)

        out = ConvLayer(
            self.conv_type, 
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=self.kernel_init,
        )(out)

        if residual.shape != out.shape:
            residual = ConvLayer(
                self.conv_type, 
                features=self.features,
                kernel_size=(1, 1),
                strides=1,
                kernel_init=self.kernel_init,
            )(residual)
        out = out + residual

        out = jnp.concatenate([out, extra_features], axis=-1) if extra_features is not None else out

        return out

class Unet(nn.Module):
    emb_features:int=64*4,
    feature_depths:list=[64, 96, 128, 160, 192],
    attention_configs:list=[{"heads":8}, {"heads":8}, {"heads":8}, {"heads":8}, {"heads":8}],
    num_res_blocks:int=1,
    num_middle_res_blocks:int=2,
    activation:Callable = jax.nn.mish

    @nn.compact
    def __call__(self, x, temb):
        temb = TimeEmbedding(self.emb_features)(temb)
        temb = TimeProjection(self.emb_features)(temb)
        # print("time embedding", temb.shape)
        feature_depths = self.feature_depths[:-1]
        attention_configs = self.attention_configs[:-1]
        x = ConvLayer(
            "conv",
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=kernel_init(1.0)
        )(x)
        downs = [x]
        # attentions = []
        for i, (features, attention_config) in enumerate(zip(feature_depths, attention_configs)):
            for j in range(self.num_res_blocks):
                x = ResidualBlock(
                    "conv", 
                    features=features,
                    kernel_init=kernel_init(1.0),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                )(x, temb, x)
                if attention_config is not None:
                    attention = nn.MultiHeadAttention(**attention_config, kernel_init=kernel_init(1), decode=False)(x)
                    x = x + attention
                    # attentions.append(attention)
                downs.append(x)
            x = Downsample(
                features=features,
                scale=2,
                activation=self.activation,
            )(x)
            if i != len(feature_depths) - 1:
                downs.append(x)
        middle_features = self.feature_depths[-1]
        middle_attention = self.attention_configs[-1]
        for j in range(self.num_middle_res_blocks):
            x = ResidualBlock(
                "separable", 
                features=middle_features,
                kernel_init=kernel_init(1.0),
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
            )(x, temb, x)
                # attentions.append(attention)
            x = ResidualBlock(
                "separable", 
                features=middle_features,
                kernel_init=kernel_init(1.0),
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
            )(x, temb, x)

        if middle_attention is not None:
            attention = nn.MultiHeadAttention(**attention_config, kernel_init=kernel_init(1), decode=False)(x)
            x = x + attention

        for i, (features, attention_config) in enumerate(zip(reversed(feature_depths), reversed(attention_configs))):
            x = Upsample(
                features=features,
                scale=2,
                activation=self.activation,
            )(x)
            for j in range(self.num_res_blocks + 1):
                x = jnp.concatenate([x, downs.pop()], axis=-1)
                x = ResidualBlock(
                    "conv", 
                    features=features,
                    kernel_init=kernel_init(1.0),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                )(x, temb, x)
                if attention_config is not None:
                    attention = nn.MultiHeadAttention(**attention_config, kernel_init=kernel_init(1), decode=False)(x)
                    x = x + attention
                    # attentions.append(attention)

        x = nn.GroupNorm(8)(x)
        x = self.activation(x)

        noise_out = ConvLayer(
            "conv",
            features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            # activation=jax.nn.mish
            kernel_init=kernel_init(0.0)
        )(x)

        # imgs_out = ConvLayer(
        #     "conv",
        #     features=3,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     kernel_init=kernel_init(0.0)
        # )(x)
        return noise_out#, attentions

# %% [markdown]
# # Training

# %%
BATCH_SIZE = 16
IMAGE_SIZE = 64

cosine_schedule = CosineNoiseSchedule(1000)
linear_schedule = LinearNoiseSchedule(1000)

# %%
import orbax.checkpoint


@struct.dataclass
class Metrics(metrics.Collection):
  loss: metrics.Average.from_output('loss') # type: ignore

class TrainState(train_state.TrainState):
    rngs:jax.random.PRNGKey
    
    def get_random_key(self):
        rngs, subkey = jax.random.split(self.rngs)
        return self.replace(rngs=rngs), subkey

class DiffusionTrainer:
    def __init__(self, 
                 model:nn.Module, 
                 optimizer: optax.GradientTransformation,
                 noise_schedule:NoiseScheduler,
                 rngs:jax.random.PRNGKey,
                 train_state:TrainState=None,
                 name:str="Diffusion",
                 load_from_checkpoint:bool=False,
                 p2_loss_weight_k:float=1,
                 p2_loss_weight_gamma:float=1,
                 ):
        self.model = model
        self.state:TrainState = None
        self.noise_schedule = noise_schedule
        self.name = name
        self.p2_loss_weights = self.noise_schedule.get_p2_weights(p2_loss_weight_k, p2_loss_weight_gamma)

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=4, create=True)
        self.checkpointer = orbax.checkpoint.CheckpointManager(self.checkpoint_path(), checkpointer, options)

        if load_from_checkpoint:
            params = self.load()
        else:
            params = None

        if train_state == None:
            self.init_state(optimizer, rngs, params=params, model=model)
        else:
            train_state.params = params
            self.state = train_state
            self.best_state = train_state

    def init_state(self, 
                   optimizer: optax.GradientTransformation, 
                   rngs:jax.random.PRNGKey,
                   params:dict=None,
                   model:nn.Module=None,
                   ):
        inp = jnp.ones((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        temb = jnp.ones((BATCH_SIZE,))
        rngs, subkey = jax.random.split(rngs)
        if params == None:
            params = model.init(subkey, inp, temb)
        self.best_loss = 1e9
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            rngs=rngs,
        )
        self.best_state = self.state

    def checkpoint_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./models'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load(self):
        step = self.checkpointer.latest_step()
        print("Loading model from checkpoint", step)
        ckpt = self.checkpointer.restore(step)
        state = ckpt['state']
        # Convert the state to a TrainState
        self.best_loss = ckpt['best_loss']
        print(f"Loaded model from checkpoint at step {step}", ckpt['best_loss'])
        return state.get('params', None)#, ckpt.get('model', None)

    def save(self, epoch=0, best=False):
        print(f"Saving model at epoch {epoch}")
        state = self.best_state if best else self.state
        # filename = os.path.join(self.checkpoint_path(), f'model_{epoch}' if not best else 'best_model')
        ckpt = {
            'model': self.model,
            'state': state,
            'best_loss': self.best_loss
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpointer.save(epoch, ckpt, save_kwargs={'save_args': save_args})

    def summary(self):
        inp = jnp.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        temb = jnp.ones((1,))
        print(self.model.tabulate(jax.random.key(0), inp, temb, console_kwargs={"width": 200, "force_jupyter":True, }))

    def _define_train_step(self):
        noise_schedule = self.noise_schedule
        model = self.model
        p2_loss_weights = self.p2_loss_weights
        @jax.jit
        def train_step(state:TrainState, batch):
            """Train for a single step."""
            images = batch
            noise_level, state = noise_schedule.generate_timesteps(images.shape[0], state)
            state, rngs = state.get_random_key()
            noise:jax.Array = jax.random.normal(rngs, shape=images.shape)
            noisy_images = noise_schedule.add_noise(images, noise, noise_level)
            def loss_fn(params):
                pred_noises = model.apply(params, noisy_images, noise_schedule.transform_steps(noise_level))
                # loss = jnp.mean(jnp.abs(pred_noises - noise))
                nloss = jnp.mean(optax.l2_loss(pred_noises.reshape((1, -1)) - noise.reshape((1, -1))), axis=1)
                nloss *= p2_loss_weights[noise_level]
                # loss = jnp.mean(optax.l2_loss(pred_noises - noise))
                nloss = jnp.mean(nloss)
                # pred_imgs = noise_schedule.remove_all_noise(noisy_images, noise, noise_level)
                # iloss = jnp.mean(jnp.abs(pred_imgs - images))
                loss = nloss# + iloss
                return loss
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads) 
            return state, loss
        return train_step
    
    def _define_compute_metrics(self):
        @jax.jit
        def compute_metrics(state:TrainState, expected, pred):
            loss = jnp.mean(jnp.square(pred - expected))
            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

    def fit(self, data, steps_per_epoch, epochs):
        data = iter(data)
        train_step = self._define_train_step()
        compute_metrics = self._define_compute_metrics()
        state = self.state
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            start_time = time.time()
            epoch_loss = 0
            with tqdm.tqdm(total=steps_per_epoch, desc=f'\t\tEpoch {epoch+1}', ncols=100, unit='step') as pbar:
                for i in range(steps_per_epoch):
                    batch = next(data)
                    state, loss = train_step(state, batch)
                    epoch_loss += loss
                    if i % 100 == 0:
                        pbar.set_postfix(loss=f'{loss:.4f}')
                        pbar.update(100)
            end_time = time.time()
            self.state = state
            total_time = end_time - start_time
            avg_time_per_step = total_time / steps_per_epoch
            avg_loss = epoch_loss / steps_per_epoch
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = state
                self.save(epoch, best=True)
            print(f"\n\tEpoch {epoch+1} completed. Avg Loss: {avg_loss}, Time: {total_time:.2f}s, Avg time per step: {avg_time_per_step:.5f}s")
        return self.state


# %%
# experiment_name = "{name}_{date}".format(
#     name="Diffusion", date=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# )
experiment_name = 'Diffusion_2024-06-09_13:04:27'
print("Experiment_Name:", experiment_name)
unet = Unet(emb_features=256, 
            feature_depths=[64, 96, 128, 192], 
            attention_configs=[{"num_heads":1, "precision":"high"}, {"num_heads":2, "precision":"high"}, {"num_heads":4, "precision":"high"}, {"num_heads":4, "precision":"high"}],
            # attention_configs=[None]*4, 
            num_res_blocks=1,
            num_middle_res_blocks=2,
            )
trainer = DiffusionTrainer(unet, optimizer=optax.adam(2e-4), 
                           noise_schedule=cosine_schedule,
                           rngs=jax.random.PRNGKey(4), 
                           name=experiment_name,
                           load_from_checkpoint=True,
                           )
#trainer.summary()
data, datalen = get_dataset("oxford_flowers102", batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE)

# %%
experiment_name = "{name}_{date}".format(
    name="Diffusion", date=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)
print("Experiment_Name:", experiment_name)
unet = Unet(emb_features=256, 
            feature_depths=[64, 96, 128, 192], 
            attention_configs=[{"num_heads":4, "precision":"default"}, {"num_heads":4, "precision":"default"}, {"num_heads":4, "precision":"default"}, {"num_heads":8, "precision":"default"}],
            # attention_configs=[None]*4, 
            num_res_blocks=1,
            num_middle_res_blocks=2,
            )
trainer = DiffusionTrainer(unet, optimizer=optax.adam(2e-4), 
                           noise_schedule=cosine_schedule,
                           rngs=jax.random.PRNGKey(4), 
                           name=experiment_name,
                        #    load_from_checkpoint=True,
                           )
#trainer.summary()

# %%
data, datalen = get_dataset("oxford_flowers102", batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE)
final_state = trainer.fit(data, datalen // BATCH_SIZE, epochs=1000)

# %%
class DiffusionSampler():   
    # Used to sample from the diffusion model
    # This is a markov chain, so we need to sample from the posterior
    def sample_step(self, noise_schedule:NoiseScheduler, current_samples, pred_noise, steps, step_size=1, clip_denoised=True, 
                    last_step=False, state:MarkovState=None) -> tuple[jnp.ndarray, MarkovState]:
        # First clip the noisy images
        pred_images = noise_schedule.remove_all_noise(current_samples, pred_noise, steps)
        # Now add some noise back to the clean images
        if last_step == True:
            return pred_images, state

        return self._renoise(noise_schedule, pred_images, current_samples, pred_noise, steps, step_size=step_size, state=state)

    def _renoise(self, noise_schedule:NoiseScheduler, reconstructed_samples, current_samples, 
                 pred_noise, steps, state:RandomMarkovState, step_size=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        return NotImplementedError

    # Used while training
    def transform_steps(self, steps):
        return steps
    
    def generate_images(self, model:nn.Module, params, 
                        noise_schedule:NoiseScheduler,
                        num_images=16, diffusion_steps=1000, 
                        priors=None, rngstate:RandomMarkovState=RandomMarkovState(jax.random.PRNGKey(42))) -> jnp.ndarray:
        if priors is None:
            rngstate, newrngs = rngstate.get_random_key()
            samples = jax.random.normal(newrngs, (num_images, IMAGE_SIZE, IMAGE_SIZE, 3))
        else:
            samples = priors
        
        step_ones = jnp.ones((num_images, ), dtype=jnp.int32)

        @jax.jit
        def sample_step(state:RandomMarkovState, samples, step, step_size):
            step = step_ones * step
            pred_noises = model.apply(params, samples, noise_schedule.transform_steps(step))
            samples, state = self.sample_step(noise_schedule, samples, pred_noises, step, state=state, step_size=step_size)
            return samples, state

        step_size = noise_schedule.max_timesteps / diffusion_steps

        for i in tqdm.tqdm(reversed(range(0, diffusion_steps))):
            samples, rngstate = sample_step(rngstate, samples, int(i * step_size), int(step_size))
        # Clip the images
        samples = noise_schedule.clip_images(samples)
        return samples

class DDPMSampler(DiffusionSampler):
    def _renoise(self, 
                 noise_schedule:DiscreteNoiseScheduler, reconstructed_samples, current_samples, 
                 pred_noise, steps, state:RandomMarkovState, step_size=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        # First estimate the q(x_{t-1} | x_t, x_0). 
        # pred_images is x_0, noisy_images is x_t, steps is t
        signal_mean = noise_schedule.posterior_mean_coef1[steps]
        noise_mean = noise_schedule.posterior_mean_coef2[steps]
        signal_mean, noise_mean = noise_schedule.reshape_rates((signal_mean, noise_mean))

        mean = signal_mean * reconstructed_samples + noise_mean * current_samples
    
        log_variance_clipped = noise_schedule.posterior_log_variance_clipped[steps].reshape(-1, 1, 1, 1)

        state, rng = state.get_random_key()
        # Now sample from the posterior
        noise = jax.random.normal(rng, reconstructed_samples.shape, dtype=jnp.float32)
        
        non_zero_mask = (1 - jnp.equal(steps, 0).astype(jnp.float32)).reshape(-1, 1, 1, 1)
        return mean + noise * jnp.exp(0.5 * log_variance_clipped) * non_zero_mask, state

class DDIMSampler(DiffusionSampler):
    def _renoise(self, 
                 noise_schedule:NoiseScheduler, reconstructed_samples, current_samples, 
                 pred_noise, steps, state:MarkovState, step_size=1) -> tuple[jnp.ndarray, MarkovState]:
        next_steps = jnp.maximum(steps - step_size, 0)
        # state, key = state.get_random_key()
        # newnoise = jax.random.normal(key, reconstructed_samples.shape, dtype=jnp.float32)
        return noise_schedule.add_noise(reconstructed_samples, pred_noise, next_steps), state

# %%
images = next(iter(data))
plotImages(images)
print(images.shape)
rng = jax.random.PRNGKey(4)
noise = jax.random.normal(rng, shape=images.shape, dtype=jnp.float32)
noise_level = 300
noise_levels = jnp.ones((images.shape[0],), dtype=jnp.int32) * noise_level
sampler = DDPMSampler()
noise_schedule = trainer.noise_schedule
noisy_images = noise_schedule.add_noise(images, noise, noise_levels)
plotImages(noisy_images)
regenerated_images = noise_schedule.remove_all_noise(noisy_images, noise, noise_levels)
plotImages(regenerated_images)
pred_noise = unet.apply(trainer.state.params, noisy_images, sampler.transform_steps(noise_levels))
reconstructed_images = noise_schedule.remove_all_noise(noisy_images, pred_noise, noise_levels)
plotImages(reconstructed_images)
samples = sampler.generate_images(trainer.model, trainer.state.params, noise_schedule, num_images=BATCH_SIZE, diffusion_steps=1000, priors=noisy_images)
plotImages(samples)

# %%
pred_noise, attentions = unet.apply(trainer.state.params, noisy_images, sampler.transform_steps(noise_levels))
reconstructed_images = noise_schedule.remove_all_noise(noisy_images, pred_noise, noise_levels)
plotImages(reconstructed_images)
bw_to_rgb = lambda x: jnp.concatenate([x, x, x], axis=-1)
plotImages(bw_to_rgb(attentions[-1][:, :, :, 0:1]))
plotImages(bw_to_rgb(attentions[-1][:, :, :, 1:2]))
plotImages(bw_to_rgb(attentions[-1][:, :, :, 2:3]))
plotImages(bw_to_rgb(attentions[-1][:, :, :, 3:4]))
plotImages(bw_to_rgb(attentions[-1][:, :, :, 4:5]))
plotImages(bw_to_rgb(attentions[-1][:, :, :, 5:6]))

# %%
plotImages(images, dpi=300)

# %%
sampler = DDIMSampler()
samples = sampler.generate_images(trainer.model, trainer.best_state.params, trainer.noise_schedule, num_images=16, diffusion_steps=50, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = DDPMSampler()
samples = sampler.generate_images(trainer.model, trainer.best_state.params, trainer.noise_schedule, num_images=16, diffusion_steps=1000, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = DDPMSampler()
samples = sampler.generate_images(trainer.model, trainer.best_state.params, trainer.noise_schedule, num_images=64, diffusion_steps=1000, priors=None)
plotImages(samples)

# %%
sampler = DDIMSampler()
samples = sampler.generate_images(trainer.model, trainer.best_state.params, trainer.noise_schedule, num_images=64, diffusion_steps=100, priors=None)
plotImages(samples)

# %%
sampler = DDPMSampler()
samples = sampler.generate_images(trainer.model, trainer.best_state.params, trainer.noise_schedule, num_images=64, diffusion_steps=1000, priors=None)
plotImages(samples)

# %%
sampler = DDIMSampler()
samples = sampler.generate_images(trainer.model, trainer.best_state.params, trainer.noise_schedule, num_images=64, diffusion_steps=100, priors=None)
plotImages(samples)

# %%
sampler = DDPMSampler()
samples = sampler.generate_images(trainer.model, trainer.state.params, noise_schedule, num_images=64, diffusion_steps=1000, priors=None)
plotImages(samples)

# %%
sampler = DDIMSampler()
samples = sampler.generate_images(trainer.model, trainer.state.params, noise_schedule, num_images=64, diffusion_steps=100, priors=None)
plotImages(samples)

# %%
samples = generate_images(trainer.state, trainer.schedule, num_images=64, diffusion_steps=1000)
plotImages(samples)

# %%


# %% [markdown]
# 


