import numpy as np

class NeuralGuide:
    def __init__(self, input_size=3, hidden_size=10):
        # Weights are the 'neural pathways'
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.learning_rate = 0.1 

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def decide_score(self, features):
        inputs = np.array(features, ndmin=2)
        # Ensure variable names match __init__ exactly
        h_out = self._sigmoid(np.dot(inputs, self.W1))
        output = np.dot(h_out, self.W2)
        return float(output[0, 0]), h_out, inputs

    def learn(self, target, pred, h_out, inputs):
        error = target - pred
        d_W2 = np.dot(h_out.T, np.array([[error]]))
        self.W2 += self.learning_rate * d_W2
        
        d_hidden = (error * self.W2.T) * (h_out * (1 - h_out))
        d_W1 = np.dot(inputs.T, d_hidden)
        self.W1 += self.learning_rate * d_W1