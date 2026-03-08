import numpy as np

from neuron import Neuron
from layer import Layer


iterations=1000

def xor_level(bits):
    # XOR ist 1, wenn die Anzahl der Einsen ungerade ist
    return [1.0,0.0]  if sum(bits) % 2 == 1 else [0.0,1.0]





inputs = np.array([1,1,1,0])
numNeurons= len(inputs)
numInputs=len(inputs)
expected = xor_level(inputs)


#lazer werden initialisiert 
l1 = Layer(numNeurons,numInputs)
l2 = Layer(numNeurons,numInputs)
l3 = Layer(numNeurons,numInputs)
l4 = Layer(2,numInputs)


print(expected)
for i in range(iterations):

	out1=l1.process(inputs)
	out2=l2.process(out1)
	out3=l3.process(out2)
	out4=l4.process(out3)

	l4.backwardsOut(expected)
	l3.backwards(l4.deltas,l4.weights)
	l2.backwards(l3.deltas,l3.weights)
	l1.backwards(l2.deltas,l2.weights)

	if i%100==0:
		print(out4)

print(expected)