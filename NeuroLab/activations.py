from .utils import *

class Activation(Layer):
    def update_parameters(self, updates):
        pass

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sigmoid = self.__call__(self.last_inputs)
        return sigmoid * (1 - sigmoid)

    def inverse(self, x):
        return np.log(x / (1 - x))

class TanH(Activation):
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2

    def inverse(self, x):
        return np.log((1 + x) / (1 - x))

class Linear(Activation):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1

    def inverse(self, x):
        return x

class ReLU(Activation):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    def inverse(self, x):
        return np.where(x > 0, x, 0)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

    def inverse(self, x):
        return np.where(x > 0, x, x / self.alpha)

class Softmax(Activation):
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True) + 1e-8
        return np.nan_to_num(self.output)

    def derivative(self, x):
      # The derivative of softmax is complex to calculate so here we will just return the grad as is.
      return 1

    def inverse(self, x):
        return np.argmax(x, axis=1)