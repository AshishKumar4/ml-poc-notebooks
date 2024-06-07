from typing import Any
from .utils import *
from .layers import *
import sklearn # Only for downloading MNIST Dataset and Accuracy Metrics

class Model:
    def __init__() -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def forward(self, x):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
    
    def predict(self, x, n_samples=1):
        raise NotImplementedError
    
    def accuracy(self, x_test, y_test) -> float:
        raise NotImplementedError
    
    def get_layers(self) -> list[Layer]:
        raise NotImplementedError

class NeuralNet(Model):
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def __call__ (self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x, n_samples=1):
        return self.__call__(x)

    def get_layers(self):
        return self.layers

    def accuracy(self, x_test, y_test):
        preds = jnp.array(antiCategorical(self.__call__(x_test)))
        expected = jnp.array(antiCategorical(y_test))
        acc = sklearn.metrics.accuracy_score(expected, preds)
        return acc
    