import numpy as np

from neuron import Neuron

class Layer:
	def __init__(self,layerSize,numInputs):
		self.layerSize=layerSize
		self.numInputs=numInputs
		self.neurons = [Neuron(self.numInputs) for i in range(self.layerSize)]

	def process(self,inputs):
		output=[]
		self.inputs=inputs
		for n in self.neurons:
			output.append(n.forward(inputs))
		return output
