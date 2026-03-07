import numpy as np

from neuron import Neuron

class Layer:
	def __init__(self,layerSize,numInputs):
		self.layerSize=layerSize
		self.numInputs=numInputs
		self.neurons = [Neuron(self.numInputs) for i in range(self.layerSize)]
		self.deltas = []


	def process(self,inputs):
		output=[]
		self.inputs=inputs
		for n in self.neurons:
			output.append(n.forward(inputs))
		return output

	def backwards(self):
		for n in self.neurons:
			n.backwards(1.0)
			self.deltas.append(n.delta.item())


inputs = np.array([-299,3,455,-123])
l= Layer(4,4)
l.process(inputs)
l.backwards()
print(l.deltas)

