import numpy as np

import plot
from neuron import Neuron
from layer import Layer


iterations=1000

def xor_level(bits):
    # XOR ist High, wenn die Anzahl der Einsen ungerade ist
    return [1.0,0.0]  if sum(bits) % 2 == 1 else [0.0,1.0]





inputs,expected = plot.generate_data(n=200)
numNeurons= 20
numInputs=len(inputs[0])



'''
lazer werden initialisiert 
Layer nimmt 
	1. zahl der neuronen
	2. form der inputs
	3. nutzt softMax ? sonst sigmoid activation
'''
input_size = 2
hidden1 = 8
hidden2 = 8
output_size = 2

l1 = Layer(hidden1, input_size, False)
l2 = Layer(hidden2, hidden1, False)
l3 = Layer(hidden1, hidden2, False)   # Softmax
l4 = Layer(output_size, hidden1, True)





for i in range(iterations):
	CurrentInput = inputs[i-1]
	currenExpected= [1.0,0.0] if expected[i]== 1  else [0.0,1.0]

	out1=l1.process(CurrentInput)
	out2=l2.process(out1)
	out3=l3.process(out2)
	out4=l4.process(out3)

	l4.backwardsOut(currenExpected)
	l3.backwards(l4.deltas,l4.weights)
	l2.backwards(l3.deltas,l3.weights)
	l1.backwards(l2.deltas,l2.weights)
	
	if i % 100 == 0:
	    print([float(x) for x in out4])


print(expected)