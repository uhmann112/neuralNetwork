import numpy as np

from neuron import Neuron
from layer import Layer

expected = 0
inputs = np.array([2,3,455,-123])
wheights =np.array([1,2,88,33])
numNeurons= len(inputs)
bias = 5
l1 = Layer(wheights,bias,numNeurons)
l2 = Layer(wheights,bias,numNeurons)
l3 = Layer(wheights,bias,numNeurons)
l4 = Layer(wheights,bias,numNeurons)

out1=l1.process(inputs)
out2=l2.process(out1)
out3=l3.process(out2)
out4=l4.process(out3)

meanPred =np.mean(out4)
error = meanPred-expected
print(meanPred)
print(error)
