import numpy as np

#import plot
from neuron import Neuron
from layer import Layer


iterations=5

def xor_level(bits):
    # XOR ist High, wenn die Anzahl der Einsen ungerade ist
    return [1.0,0.0]  if sum(bits) % 2 == 1 else [0.0,1.0]











'''
lazer werden initialisiert 
Layer nimmt 
	1. zahl der neuronen
	2. form der inputs
	3. nutzt softMax ? sonst sigmoid activationfunktion
'''
input_size = 2
hidden1 = 8
hidden2 = 8
output_size = 2

l1 = Layer(hidden1, input_size, False)
l2 = Layer(hidden2, hidden1, False)
l3 = Layer(hidden1, hidden2, False)   # Softmax
l4 = Layer(output_size, hidden1, True)


def predict(point):
    o1 = l1.process(point)
    o2 = l2.process(o1)
    o3 = l3.process(o2)
    o4 = l4.process(o3)
    return o4

def runNetwork(data):	
	for i in range(iterations):
		data=np.array(data,dtype=float)
		for x in data:
			CurrentInput = x[:2]
			expected = [1.0, 0.0] if x[2] == 1 else [0.0, 1.0]

			out1=l1.process(CurrentInput)
			out2=l2.process(out1)
			out3=l3.process(out2)
			out4=l4.process(out3)

			l4.backwardsOut(expected)
			l3.backwards(l4.deltas,l4.weights)
			l2.backwards(l3.deltas,l3.weights)
			l1.backwards(l2.deltas,l2.weights)
		if iterations %10 ==0:
			print(out4)
			

	print(expected)


