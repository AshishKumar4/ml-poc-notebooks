# import numpy as np
import cupy as np

def antiCategorical(arr):
    return np.argmax(arr, axis=1)

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
    if np.any(np.isnan(arr)):
        print('NaN found in arr of shape {arr}, {name} {val}'.format(arr=arr.shape, name=name, val=arr))
        raise ValueError('NaN found in arr of shape {arr}, {name}'.format(arr=arr.shape, name=name))
