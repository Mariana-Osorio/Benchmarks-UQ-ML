import numpy as np

def model(X, P):
	X = np.array(X, ndmin=2)
	result = np.sin(X[:, 0]) + P['a'] * np.power(np.sin(X[:, 1]), 2) + P['b'] * np.power(X[:,2], 4) * np.sin(X[:, 0])
	
	return result