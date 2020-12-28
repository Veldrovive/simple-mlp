import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class Activation(Enum):
    RELU = 1
    SIGMOID = 2

class Layer:
    def __init__(self, in_size: int, out_size: int, activation: Activation = Activation.RELU):
        self.in_size = in_size
        self.out_size = out_size

        self.activation = activation

        # We add one extra column for the bias
        self.weights = np.random.rand(out_size, in_size + 1) - np.full((out_size, in_size + 1), 0.5)
        self.bias = np.random.rand(out_size) - np.full(out_size, 0.5)

        self.previous_layer_activations = None
        self.last_input = None
        self.last_output = None
        self.deltas = None

    def out(self, last_layer_out: np.ndarray):
        self.previous_layer_activations = last_layer_out
        out = np.matmul(self.weights, np.concatenate((np.array([1]), last_layer_out)))  # We concat one for the bias
        self.last_input = out
        out = self.apply_activation(out)
        self.last_output = out
        return out

    def compute_deltas(self, own_out, next_deltas):
        activation_derivatives = own_out(1-own_out).reshape(self.out_size)
        cost_function_derivatives = np.matmul(self.weights, next_deltas).reshape(self.out_size)
        new_deltas = activation_derivatives * cost_function_derivatives
        return new_deltas

    def error(self, target):
        return (1/(2*self.out_size)) * np.sum((self.last_output - target)**2)

    def apply_activation(self, vector: np.ndarray):
        original_shape = vector.shape
        if self.activation == Activation.RELU:
            vector = vector*(vector > 0)
        elif self.activation == Activation.SIGMOID:
            vector = 1 / (1 + np.exp(-1*vector))
        vector.reshape(original_shape)
        return vector

    def apply_activation_derivative(self, vector: np.ndarray):
        original_shape = vector.shape
        if self.activation == Activation.RELU:
            vector = (vector > 0).astype(int)
        elif self.activation == Activation.SIGMOID:
            vector = vector*(1 - vector)
        vector.reshape(original_shape)
        return vector

    def update(self, next_layer = None, target = None, learning_rate=0.5):
        self.deltas = None
        if self.last_input is None:
            raise RuntimeError("Forward pass must be completed before backprop")
        current_activation_derivative = self.apply_activation_derivative(self.last_output)
        if next_layer is None:
            self.deltas = np.transpose(((1/self.out_size) * (self.last_output - target) * current_activation_derivative)[np.newaxis])
        else:
            next_weights_no_bias = next_layer.weights[:, 1:]
            next_deltas = next_layer.deltas
            next_error = np.matmul(next_weights_no_bias.transpose(), next_deltas)
            # self.deltas = np.transpose((np.dot(next_error, current_activation_derivative[np.newaxis])))
            self.deltas = next_error * current_activation_derivative[np.newaxis].transpose()
        differences = learning_rate * np.matmul(self.deltas, np.concatenate((np.array([1]), self.previous_layer_activations))[np.newaxis])
        self.weights = self.weights - differences



class Network:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.input = []
        self.errors = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, data: np.ndarray):
        self.activations = []
        self.input = data
        x = data
        for layer in self.layers:
            x = layer.out(x)
            self.activations.append(x)
        return x

    def return_reset_error(self):
        mean = np.mean(self.errors)
        self.errors = []
        return mean

    def backward(self, ground_truth: np.ndarray, learning_rate=0.5):
        layer = self.layers[-1]
        layer.update(target=ground_truth, learning_rate=learning_rate)
        for previous_layer in reversed(self.layers[:-1]):
            previous_layer.update(next_layer=layer, learning_rate=learning_rate)
            layer = previous_layer
        self.errors.append(self.layers[-1].error(ground_truth))


def example_train(train_in, train_out, network: Network, delta_cutoff=-1e-6, error_cutoff=0.01):
    data = list(zip(train_in, train_out))
    possible_datapoints = list(range(len(train_in)))
    errors = [-np.inf]
    error_deltas = []
    lr_max = 0.55
    lr_min = 0.02
    num_epochs = 30000
    for i in range(num_epochs):
        lr = ((lr_min - lr_max) / num_epochs)*i + lr_max
        print("Epoch", i, "lr:", lr)
        np.random.shuffle(possible_datapoints)
        for j in possible_datapoints:
            network_input, network_output = data[j]
            output = network.forward(network_input)
            net.backward(network_output, learning_rate=lr)
            print(network_input,"->",network_output, output)
        error = network.return_reset_error()
        error_delta = error - errors[-1]
        errors.append(error)
        error_deltas.append(error_delta)
        print("Error:", error, "Delta", error_delta)
        print("\n")
        if i > 50 and error < error_cutoff and np.mean(error_deltas[-50:]) > delta_cutoff:
            break
    plt.plot(errors)
    plt.savefig("errors.png")

    

if __name__ == "__main__":
    np.random.seed(0)

    net = Network()
    net.add_layer(Layer(2, 4, Activation.RELU))
    net.add_layer(Layer(4, 2, Activation.RELU))
    net.add_layer(Layer(2, 1, Activation.SIGMOID))

    # test_data = np.array([5, 3])
    # out = net.forward(test_data)
    # print(out)

    xor_in = [
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1])
    ]

    xor_out = [
        np.array([0]),
        np.array([1]),
        np.array([1]),
        np.array([0])
    ]

    # output = net.forward(xor_in[0])
    # print("Output:", output)
    # net.backward(xor_out[0], learning_rate=0.5)
    # output = net.forward(xor_in[0])
    # print("New Output:", output)

    # for i in range(100):
    #     output = net.forward(xor_in[0])
    #     net.backward(xor_out[0], learning_rate=0.5)
    #     new_output = net.forward(xor_in[0])
    #     print(i, "- Old:", output, "New:", new_output)

    example_train(xor_in, xor_out, net)

    # l = Layer(2, 10)
    # print(l.out(np.array([1, 2])))

