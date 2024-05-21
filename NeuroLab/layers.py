from .utils import *
import math

class Dense(Layer):
    def __init__(self, n_input, n_output, use_bias=True):
        self.weights = np.random.normal(0, scale=(1/float(math.sqrt(n_input))), size=(n_input, n_output)).astype(np.float64)
        self.biases = np.zeros((1, n_output)).astype(np.float64)
        self.use_bias = use_bias

    def __call__(self, x):
        output = np.dot(x, self.weights)
        if self.use_bias == True:
            output += self.biases
        return output

    # derivative of the layer w.r.t the output of the layer
    def derivative(self, x):
        return self.weights

    def inverse(self, x):
        inv_weight = np.linalg.pinv(self.weights)
        x = np.nan_to_num(np.dot(x, inv_weight))
        return x

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(self.last_inputs.T, grad_output)
        grad_biases = np.mean(grad_output, axis=0, keepdims=True)

        return grad_input, (grad_weights, grad_biases)

    def update_parameters(self, updates):
        delta_weights, delta_biases = updates
        self.weights -= delta_weights
        self.biases -= delta_biases