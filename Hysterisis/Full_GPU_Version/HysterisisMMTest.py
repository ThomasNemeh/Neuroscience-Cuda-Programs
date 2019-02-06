import HysterisisMM as hys
import numpy as np
'''Simple one neuron network with self-excitatory weights and external inputs'''
#list of neurons
V = np.array([0])
'''print("V.shape " + str(V.shape[0]))
print("V[0] " + str(V[0]))
print("V: " + str(V))'''

#list of weight connections
W = np.array([2])
'''print("W.shape: " + str(W.shape[0]))
print("W[0]: " + str(W[0]))
print("W: " + str(W))'''

#external inputs
ext_input = [0] * 30
ext_input.extend([1]*40)
ext_input.extend([0.5]*30)
E = np.ndarray(shape=(100,), dtype=float, buffer=np.asarray(ext_input))
'''print("E.shape " + str(E.shape[0]))
print("E: " + str(E))'''


timestep = .1

#print("Timestep: " + str(timestep))

x = hys.hysterisis(100, timestep, 0, 4, .5, layers=V, weights=W, external=E)
hys.printLastResults()
#print("Final Result: " + str(x))

#*************************************************************************
#3 neuron network with no external inputs

V_2 = np.array([1, 1, 0])

W_2 = np.array([[0, 2.5, 1], 
			 [1, 0, 2],
			 [0, 0, 2]])
			 
#print(str(W_2))
			 
y = hys.hysterisis(100, timestep, 0, 4, .5, layers=V_2, weights=W_2)
#print("Final Result: " + str(y))
hys.printLastResults()

#**************************************************************************
#3 neuron network random initial activation values and weight connections

z = hys.hysterisis(100, timestep, 1, 4, .5, num_neurons=3)
hys.printLastResults()
#print("Final Result: " + str(z))




