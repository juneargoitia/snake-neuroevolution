import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=8, hidden_size=16, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(-1, 1, (self.hidden_size,))
        self.w2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(-1, 1, (self.output_size,))

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, inputs):
        x = np.array(inputs, dtype=float)
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2

        return int(np.argmax(z2))

    def get_weights(self):
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])

    def set_weights(self, flat_weights):
        idx_w1 = self.input_size * self.hidden_size
        idx_b1 = idx_w1 + self.hidden_size
        idx_w2 = idx_b1 + (self.hidden_size * self.output_size)

        self.w1 = flat_weights[:idx_w1].reshape((self.input_size, self.hidden_size))
        self.b1 = flat_weights[idx_w1:idx_b1]
        self.w2 = flat_weights[idx_b1:idx_w2].reshape((self.hidden_size, self.output_size))
        self.b2 = flat_weights[idx_w2:]