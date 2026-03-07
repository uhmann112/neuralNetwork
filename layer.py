import numpy as np

from neuron import Neuron

class Layer:
	def __init__(self,weights,bias,layerSize):
		self.weights=weights
		self.bias=bias
		self.layerSize=layerSize

	def process(self,inputs):
		output=[]
		neurons = [Neuron(self.weights,self.bias) for i in range(self.layerSize)]
		for n in neurons:
			output.append(n.forward(inputs))
		return output