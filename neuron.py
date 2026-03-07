import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidDerivative(prevOut):
	s = sigmoid(prevOut)
	return s * (1 - s)

def delta(exepected):
	return (prevOut-expected)*sigmoidDerivative




class Neuron:
	def __init__(self,numInputs,lr=0.1):
		self.weights=np.random.rand(1,numInputs)
		self.bias = np.random.randint(5)
		self.lr=lr

	def forward(self,inputs):
		self.inputs = inputs
		self.weightedSum= np.dot(self.weights,self.inputs)+self.bias
		self.output = sigmoid(self.weightedSum).item()
		return self.output

	def backwards(self, expected):
		self.delta = (self.output - expected) * sigmoidDerivative(self.weightedSum)
		self.weights -= self.lr * self.delta * self.inputs
		self.bias -= self.lr * self.delta

