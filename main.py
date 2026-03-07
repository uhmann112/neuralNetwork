import numpy as np

from neuron import Neuron
from layer import Layer


iterations=200000

expected = 1
inputs = np.array([29999,3,455,-123])
wheights =np.array([1,2,88,33])
numNeurons= len(inputs)
numInputs=len(inputs)

bias = 5

#lazer werden initialisiert 
l1 = Layer(numNeurons,numInputs)
l2 = Layer(numNeurons,numInputs)
l3 = Layer(numNeurons,numInputs)
l4 = Layer(numNeurons,numInputs)



for i in range(iterations):

	out1=l1.process(inputs)
	out2=l2.process(out1)
	out3=l3.process(out2)
	out4=l4.process(out3)

	meanPred =np.mean(out4)
	error = meanPred-expected
print(f"loop ran {iterations} times succesfully!")
print(f"mean prediction : {meanPred}")
print(f"expected was {expected}")
print(f"fehler ist {error}")
