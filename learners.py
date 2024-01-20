import numpy as np
from FFNN import FFNN

class FFNNLearner:
    def __init__(self, input_size, output_size, hidden_sizes, data, learning_rate=0.99, learning_decay=0.995):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.nn = FFNN(input_size, output_size, hidden_sizes)
        self.data = data
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
    
    def train(self, X, Y, epochs=1, verbose_freq=0):
        self.nn.train(X, Y, self.learning_rate, self.learning_decay, epochs, verbose_freq)
    
    def test(self, X):
        return self.nn.test(X)

    def save_model(self, filename):
        self.nn.save_model(filename)

    def load_model(self, filename):
        self.nn.load_model(filename)
    

class LobsterLearner:
    def __init__(self, input_size, output_size, hidden_sizes, data, gamma, learning_rate=0.99, learning_decay=0.995, floating_cost=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.nn = FFNN(input_size, output_size, hidden_sizes)
        self.target = FFNN(input_size, output_size, hidden_sizes)
        self.data = data
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.floating_cost = floating_cost
        self.last_action = 1
        self.index = 0

    def train(self, X, trips=1):
        for i in range(trips):
            self._reset()
            self.target.weights = self.nn.weights.copy()
            for i in range(X.shape[0]):
                state = self._calc_state(i, self.last_action)
                pred = self.nn.train(state, self._get_Y(), self.learning_rate, 1, 1)
                self.last_action = pred.argmax()
                self.index += 1
            self.learning_rate *= self.learning_decay
    
    def test(self, index, last_action=None):
        if last_action == None:
            last_action = self.last_action
        X = self._calc_state(index, last_action)
        self.last_action = self.nn.test(X).argmax()
        return self.last_action
    
    def save_model(self, filename):
        self.nn.save_model(filename)
    
    def load_model(self, filename):
        self.nn.load_model(filename)

    def _calc_reward(self, action):
        if self.index >= self.data.shape[0] - 1:
            return 0
        cost = 0
        if self.last_action != action:
            # rewards normalized to buying 1 share, so floating cost doesn't differ
            cost = self.floating_cost
        reward = (self.data['price'].iloc[self.index + 1] - self.data['price'].iloc[self.index]) * (action - 1) - cost
        return reward
    
    def _calc_state(self, index, action):
        action -= 1
        if index >= self.data.shape[0] - 1:
            return np.array([[1, action]])
        vol0 = self.data['b_size_0'].iloc[index] / self.data['a_size_0'].iloc[index]
        # vol1 = self.data['b_size_1'].iloc[index] / self.data['a_size_1'].iloc[index]
        # vol2 = self.data['b_size_2'].iloc[index] / self.data['a_size_2'].iloc[index]
        if vol0 > 100:
            vol0 = 100
        return np.array([[vol0, action]])
    
    def _reset(self):
        self.last_action = 1
        self.index = 0

    def _get_Y(self):
        # Y is the next indexes price
        Y = np.zeros(self.output_size)
        for i in range(self.output_size):
            future = self.target.test(self._calc_state(self.index + 1, i)).max()
            r = self._calc_reward(i)
            Y[i] = (r + self.gamma * future)
        return np.array([Y])