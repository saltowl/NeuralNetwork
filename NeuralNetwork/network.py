import numpy as np
from sklearn.metrics import mean_squared_error

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class NeuralNetwork:
	def __init__(self, x, y, num_on_hidden=2):
		self.input = x
		self.y = y
		self.output = np.zeros(self.y.shape)

		self.weights1 = 2 * np.random.rand(self.input.shape[1], num_on_hidden) - 1
		self.weights2 = 2 * np.random.rand(num_on_hidden, 1) - 1

	def feedforward(self, X):
		self.layer1 = sigmoid(np.dot(X, self.weights1))
		self.output = sigmoid(np.dot(self.layer1, self.weights2))

	def backprop(self, learning_rate):
		l2_error = self.output - self.y
		d_weights2 =  l2_error * sigmoid((self.output), derivative=True)
		
		l1_error = d_weights2.dot(self.weights2.T)
		d_weights1 = l1_error * sigmoid((self.layer1), derivative=True)

		self.weights2 -= self.layer1.T.dot(d_weights2 * learning_rate)
		self.weights1 -= self.input.T.dot(d_weights1 * learning_rate)
	
	def train(self, num_epochs, learning_rate, display_loss):
		for i in range(num_epochs):
			self.feedforward(self.input)
			if display_loss:
				print("Epoch " + str(i) + ': ' + '; MSE: ' + str(mean_squared_error(self.y, self.output)))
			self.backprop(learning_rate)

	def test(self, X):
		self.feedforward(X)

		return self.output[:len(X)]

