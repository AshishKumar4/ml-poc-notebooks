from .utils import *

class MeanSquaredError:
    @staticmethod
    def forward(predictions, targets):
        loss = np.mean(0.5 * (predictions - targets)**2)
        # print("Loss=>", loss)
        return loss

    @staticmethod
    def backward(predictions, targets):
        # return ((2 * (predictions - targets)) / predictions.size)# * np.ones_like(predictions)
        return predictions-targets

class SoftmaxCrossEntropy:
    @staticmethod
    def forward(predictions, targets):
        return np.nan_to_num(-np.mean(targets * np.log(predictions)) - np.mean((1 - targets) * np.log(1 - predictions)))

    @staticmethod
    def backward(predictions, targets):
        return (predictions - targets) / predictions.size
