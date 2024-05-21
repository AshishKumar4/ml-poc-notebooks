from .utils import *
import math

class Dense(Layer):
    def __init__(self, n_input, n_output, use_bias=True):
        self.weights = jax.random.normal(get_random_key(), (n_input, n_output), jnp.float64) * (1/float(math.sqrt(n_input)))
        self.biases = jnp.zeros((1, n_output)).astype(jnp.float64)
        self.use_bias = use_bias

    def __call__(self, x):
        output = jnp.dot(x, self.weights)
        if self.use_bias == True:
            output += self.biases
        return output

    # derivative of the layer w.r.t the output of the layer
    def derivative(self, x):
        return self.weights

    def inverse(self, x):
        inv_weight = jnp.linalg.pinv(self.weights)
        x = jnp.nan_to_num(jnp.dot(x, inv_weight))
        return x

    def backward(self, grad_output):
        grad_input = jnp.dot(grad_output, self.weights.T)

        grad_weights = jnp.dot(self.last_inputs.T, grad_output)
        grad_biases = jnp.mean(grad_output, axis=0, keepdims=True)

        return grad_input, (grad_weights, grad_biases)

    def update_parameters(self, updates):
        delta_weights, delta_biases = updates
        self.weights -= delta_weights
        self.biases -= delta_biases