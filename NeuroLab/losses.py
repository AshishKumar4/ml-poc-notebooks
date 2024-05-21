from .utils import *

class MeanSquaredError:
    @staticmethod
    @jax.jit
    def forward(predictions, targets):
        loss = jnp.mean(0.5 * (predictions - targets)**2)
        # print("Loss=>", loss)
        return loss

    @staticmethod
    @jax.jit
    def backward(predictions, targets):
        # return ((2 * (predictions - targets)) / predictions.size)# * jnp.ones_like(predictions)
        return predictions-targets

class SoftmaxCrossEntropy:
    @staticmethod
    @jax.jit
    def forward(predictions, targets):
        return jnp.nan_to_num(-jnp.mean(targets * jnp.log(predictions)) - jnp.mean((1 - targets) * jnp.log(1 - predictions)))

    @staticmethod
    @jax.jit
    def backward(predictions, targets):
        return (predictions - targets) / predictions.size
