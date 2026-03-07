import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class Neuron:
	def __init__(self,weights,bias):
		self.weights=weights
		self.bias=bias

	def forward(self,inputs):
		total= np.dot(self.weights,inputs)
		return sigmoid(total)



