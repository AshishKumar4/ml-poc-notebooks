# import numpy as np
import jax.numpy as jnp
import jax

global RANDOM_KEY
RANDOM_KEY = jax.random.PRNGKey(4)

def get_random_key():
    global RANDOM_KEY
    RANDOM_KEY, subkey = jax.random.split(RANDOM_KEY)
    return subkey

def set_random_key(seed):
    global RANDOM_KEY
    RANDOM_KEY = jax.random.PRNGKey(seed)

def antiCategorical(arr):
    return jnp.argmax(arr, axis=1)