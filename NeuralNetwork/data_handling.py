import numpy as np
import pandas as pd

def generate_data_for_first_func():
	x = [0, 0, 1, 1]
	y = [0, 1, 0, 1]

	for i in range(5):
		x.extend(x)
		y.extend(y)

	output = [int(not(x[i] or y[i]) or not(y[i]) or (x[i] and y[i])) for i in range(len(x))]

	df = pd.DataFrame({'x':x, 'y':y, 'F':output})
	np.random.shuffle(df.values)
	df.to_csv('../data/data1.csv', index=False)

def generate_data_for_second_func():
	x = [0, 0, 0, 0, 1, 1, 1, 1]
	y = [0, 0, 1, 1, 0, 0, 1, 1]
	z = [0, 1, 0, 1, 0, 1, 0, 1]

	for i in range(5):
		x.extend(x)
		y.extend(y)
		z.extend(z)
	
	output = [int((not x[i] or not y[i]) and (not x[i] or z[i])) for i in range(len(x))]

	df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'F':output})
	np.random.shuffle(df.values)
	df.to_csv('../data/data2.csv', index=False)

def read_data():
	data = []
	for i in range(2):
		data.append(pd.read_csv('../data/data' + str(i + 1) + '.csv'))

	return data
