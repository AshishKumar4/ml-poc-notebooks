from .utils import *
from .models import *
from .layers import *

class Optimizer:
    def __init__(self, model: Model, loss):
        self.model = model
        self.loss = loss

    def train_step(self, x_batch, y_batch):
        print("optimizer train_step")
        raise NotImplementedError

    def fit(self, train_data, test_data, epochs, batch_size=128, verbose=True, **kwargs):
        x_train, y_train = train_data
        x_test, y_test = test_data
        n_batches = max(1, len(x_train) // batch_size)

        self.on_train_start(**kwargs)

        for epoch in range(epochs):
            shuffle_indices = jax.random.permutation(get_random_key(), len(x_train))
            x_train = x_train[shuffle_indices]
            y_train = y_train[shuffle_indices]

            self.on_epoch_start(epoch)

            for i in range(n_batches):
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]

                loss_value = self.train_step(x_batch, y_batch)

            if verbose:
                acc = self.model.accuracy(x_test, y_test)
                self.on_reporting(epoch, loss_value, acc)

            self.on_epoch_end(epoch)


    def on_reporting(self, epoch, loss_value, acc):
        print(f'Epoch {epoch}, Loss: {loss_value}, test_acc: {acc}')

    def on_epoch_end(self, epoch):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_train_start(self, **kwargs):
        pass

class SGD_Optimizer(Optimizer):
    def __init__(self, model: Model, learning_rate, loss, gamma=1, delta=2):
        super().__init__(model, loss)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.delta = delta

    def train_step(self, x_batch, y_batch):
        predictions = self.model.forward(x_batch)
        loss_value = self.loss.forward(predictions, y_batch)
        grad_loss = self.loss.backward(predictions, y_batch)

        for layer in reversed(self.model.get_layers()):
            # Back propogate the loss to the layer
            grad_loss, deltas = layer.backward(grad_loss)
            if deltas != None:
                # Simple Stochastic Gradient Decent
                delta_weights, delta_biases = deltas
                deltas = (delta_weights * self.learning_rate, delta_biases * self.learning_rate)
            # Update the weights
            layer.update_parameters(deltas)

        return loss_value

    def on_epoch_end(self, epoch):
        # Decay Weight
        self.learning_rate *= self.gamma
        self.delta *= self.gamma

    # import cupy as np

class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, model, learning_rate, loss):
        super().__init__(model, loss)
        self.learning_rate = learning_rate
        self.best_loss = float("inf")

    def forward_with_perturbations(self, x_train, stddev):
        perturbations = []
        inputs = x_train
        for layer in self.model.get_layers():
            if isinstance(layer, SimulatedAnnealingLayer):
                perturbation = layer.perturb_parameters(stddev)
                perturbations.append(perturbation)
                inputs = layer.forward_with_perturbations(inputs, perturbation)
            else:
                inputs = layer.forward(inputs)
        return inputs, perturbations

    def update_parameters(self, perturbations):
      i = 0
      for layer in self.model.get_layers():
            if isinstance(layer, SimulatedAnnealingLayer):
                layer.update_parameters(perturbations[i])
                i += 1

    def cooling_schedule(self, old_loss, new_loss, current_temp, current_step_size, cooling_rate, step_decay_rate=0.9999):
        deltaLoss = new_loss - old_loss
        deltaLossScale = deltaLoss / new_loss

        if abs(deltaLossScale) > min(1e-4, current_temp):
            scalefactor = min(1, current_temp/new_loss)
            elastic_rate = 1.0 - scalefactor
            elastic_rate = (elastic_rate + scalefactor) / 2
            rate = max(cooling_rate, elastic_rate)
            new_temp = current_temp * rate
            # new_temp = current_temp * cooling_rate
            new_step_size = current_step_size * step_decay_rate
        else:
            new_temp = current_temp / cooling_rate
            new_step_size = self.learning_rate
            self.learning_rate *= step_decay_rate

        return new_temp, new_step_size

    def train_step(self, x_batch, y_batch):
        for j in range(self.sample_per_batch):
            predictions, perturbations = self.forward_with_perturbations(x_batch, self.step_size)
            loss = self.loss.forward(predictions, y_batch)
            delta_E = (loss - self.best_loss)
            energy_cost = jnp.exp(-delta_E / self.current_temp)
            acceptance_rate = energy_cost if delta_E > 0 else  1.0
            if delta_E < 0 or jax.random.uniform(get_random_key()) <= acceptance_rate:
                # Accept proposal
                self.update_parameters(perturbations)
                self.best_loss = loss
                # print(f'Accepted proposal due to temperature {loss} exp: {acceptance_rate}, deltaE {delta_E}')
        return self.best_loss

    def on_epoch_start(self, epoch):
        self.old_loss = self.best_loss

    def on_epoch_end(self, epoch):
        self.current_temp, self.step_size = self.cooling_schedule(self.old_loss, self.best_loss, self.current_temp, self.step_size, self.cooling_rate, 0.9999)

    def on_reporting(self, epoch, loss_value, acc):
        print(f'Epoch {epoch}, BestLoss: {loss_value}, Temperature {self.current_temp}, step_size {self.step_size}, test_acc: {acc}')

    def on_train_start(self, sample_per_batch=1, initial_temp=1.0, cooling_rate=0.99):
        self.current_temp = initial_temp
        # self.best_loss = float("inf")
        self.step_size = self.learning_rate
        self.sample_per_batch = sample_per_batch
        self.cooling_rate = cooling_rate

    # Extreme Learning Machines
class ELM_Optimizer(Optimizer):
    def __init__(self, model, learning_rate, loss):
        super().__init__(model, loss)
        self.learning_rate = learning_rate

    def train_step(self, x_batch, y_batch, alpha=0):
        predictions = self.model.forward(x_batch)
        loss_value = self.loss.forward(predictions, y_batch)

        # Smooth out y_batch
        y_batch = jnp.where(y_batch > 0.5, 0.9, 0.1)
        expected = y_batch
        for layer in reversed(self.model.get_layers()):
            if isinstance(layer, Dense):
                x_inv = jnp.linalg.pinv(layer.last_inputs)
                weight_approx = jnp.dot(x_inv, expected)
                layer.weights = layer.weights * alpha + weight_approx * (1 - alpha)
                # layer.weights = weight_approx
                # expected = layer.last_inputs
                expected = layer.inverse(expected)
            else:
                expected = layer.inverse(expected)
        return loss_value

