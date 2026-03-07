import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class Neuron:
	def __init__(self,numInputs):
		self.weights=np.random.rand(1,numInputs)
		self.bias = np.random.randint(5)

	def forward(self,inputs):
		total= np.dot(self.weights,inputs)+self.bias
		return sigmoid(total).item()

