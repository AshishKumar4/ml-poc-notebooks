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

class Layer:
    # Actual operation of the layer
    def __call__(self, inputs):
        raise NotImplementedError

    # Derivative of the layer w.r.t the output
    def derivative(self, x):
        raise NotImplementedError

    def forward(self, x):
        self.last_inputs = x
        return self.__call__(x)

    def backward(self, grad_output):
        return self.derivative(self.last_inputs) * grad_output, None

    def inverse(self, x):
        raise NotImplementedError

    def update_parameters(self, updates):
        raise NotImplementedError

def checkNan(arr, name):
    if jnp.any(jnp.isnan(arr)):
        print('NaN found in arr of shape {arr}, {name} {val}'.format(arr=arr.shape, name=name, val=arr))
        raise ValueError('NaN found in arr of shape {arr}, {name}'.format(arr=arr.shape, name=name))
