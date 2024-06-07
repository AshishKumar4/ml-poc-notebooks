from .utils import *
import math

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

class SimulatedAnnealingLayer(Layer):
    def forward_with_perturbations(self, inputs, perturbations):
        raise NotImplementedError

    def perturb_parameters(self, stddev = 0.01):
        raise NotImplementedError

class SADense(SimulatedAnnealingLayer, Dense):
    def __init__(self, n_input, n_output, *args, **kwargs):
        super().__init__(n_input, n_output, *args, **kwargs)

    def forward_with_perturbations(self, inputs, perturbations):
        delta_w, delta_b = perturbations
        self.inputs = inputs
        self.output = jnp.dot(inputs, self.weights + delta_w) + self.biases + delta_b
        return self.output

    def perturb_parameters(self, stddev = 0.1):
        """Randomly perturbs weights and biases with a given standard deviation."""
        perturbation_weights = jax.random.normal(get_random_key(), self.weights.shape) * stddev
        perturbation_biases = jax.random.normal(get_random_key(), self.biases.shape) * stddev
        return perturbation_weights, perturbation_biases

    def update_parameters(self, updates):
        delta_weights, delta_biases = updates
        self.weights += delta_weights
        self.biases += delta_biases