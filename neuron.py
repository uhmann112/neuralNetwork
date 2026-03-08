import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class Neuron:
	def __init__(self,numInputs,lr=0.1):
		self.weights=np.random.rand(numInputs)
		self.bias = np.random.randint(5)
		self.lr=lr
		#print(f"prev Wheights:{self.weights}")

	def forward(self,inputs):
		self.inputs = inputs
		self.weightedSum= np.dot(self.weights,self.inputs)+self.bias
		self.output = sigmoid(self.weightedSum).item()
		return self.output

	def backwards(self, delta):
		self.delta = delta
		self.weights -= self.lr * self.delta * self.inputs
		self.bias -= self.lr * self.delta
		#print(f"updated Wheights:{self.weights}")



