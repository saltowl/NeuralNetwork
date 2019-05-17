import data_handling as dh
import numpy as np
from network import NeuralNetwork
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def bool_func(data, num_on_hidden=2, num_epochs=200, learning_rate=0.1, display_loss=False, label=''):
	x = data.drop(['F'], axis=1).values
	y = np.array([data['F'].values]).T
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

	nw = NeuralNetwork(x_train, y_train, num_on_hidden)
	loss = nw.train(num_epochs, learning_rate, display_loss, label)
	pred = nw.test(x_test)
	pred = (pred > 0.5).astype("int").ravel()
	score = accuracy_score(y_test, pred) * 100

	print_info(data, loss, num_on_hidden, num_epochs, learning_rate, score)

def print_info(data, loss, num_on_hidden, num_epochs, learning_rate, score):
	print("Number of neurons: " + str(data.shape[1] + num_on_hidden))
	print("Count of epochs: " + str(num_epochs))
	print("MSE at the last epoch: " + str(round(loss, 5)))
	print("Learning rate: " + str(learning_rate))
	print("Accuracy score on test data: " + str(score) + '%\n')

def main():
	np.random.seed(1)
	plt.style.use("bmh")

	dh.generate_data_for_first_func()
	dh.generate_data_for_second_func()

	data = dh.read_data()

	first_label = 'First FFN'
	print(first_label)
	bool_func(data[0], num_on_hidden=4, num_epochs=80, label=first_label)

	second_label = 'Second FFN'
	print(second_label)
	bool_func(data[1], num_on_hidden=8, num_epochs=50, label=second_label)

	plt.show()


main()