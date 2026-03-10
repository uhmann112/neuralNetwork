import numpy as np

from neuron import Neuron


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softMax(ar):
    e = np.exp(ar - np.max(ar))  # np.max für numerische Stabilität
    return e / e.sum()

class Layer:
	def __init__(self,layerSize,numInputs,isSoftMax):
		self.layerSize=layerSize
		self.numInputs=numInputs
		self.neurons = [Neuron(self.numInputs) for i in range(self.layerSize)]
		self.deltas = []
		self.isSoftMax=isSoftMax



	def process(self,inputs):
		self.output=[]
		self.inputs=inputs
		for n in self.neurons:
			self.output.append(n.forward(inputs))
		self.weights = np.array([n.weights.copy() for n in self.neurons])
		if self.isSoftMax==True:
			return np.array(softMax(self.output))
		else:
			return np.array(self.output)


	def backwardsOut(self,expected):
		self.expected=np.array(expected)
		self.deltas=[]
		self.prevWeights = [n.weights.copy() for n in self.neurons]

		self.deltas = (self.output-self.expected)
		for i, n in enumerate(self.neurons):
			n.backwards(self.deltas[i])

	def backwards(self,deltasNext,weightsNext):
		z = np.array([n.weightedSum for n in self.neurons])
		self.derivative = np.array([n.output * (1 - n.output) for n in self.neurons])
		self.influence = np.dot(weightsNext.T, deltasNext)
		self.deltas = self.influence * self.derivative
		for i, n in enumerate(self.neurons):
			n.backwards(self.deltas[i])



