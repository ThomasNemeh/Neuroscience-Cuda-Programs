### Script that implements a basic neural network with hysteresis
### W defines the n by n connection matrix between n layers of neurons
### V describes the inital activities of n layers of neurons (each with one neuron for now)

import sys
sys.path.insert(0,"/home/psimen/anaconda3/lib/python3.6/site-packages")                                                                                                      

from libc.stdlib cimport free, malloc
from cpython cimport PyObject, Py_INCREF

np.import_array()

cimport numpy as np
import numpy as np
import matplotlib.pyplot as plt

cdef extern from "MatrixMultiplication.h":
    void matrixMultiplication(float *layers, float *weights, int dim, float L, float M);

def sigmoid(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta))) ### plot this function

class HysteresisNetwork:
    #cdef np.ndarray layers
    #cdef np.ndarray weights
	
    def __init__(self, w, v, lam, beta, dt, ext_input):
        self.layers    = v
        self.weights   = w 
        self.ext_input = ext_input
        self.timestep  = dt
        self.lam = lam;
        self.beta = beta;
        self.output = np.zeros(self.layers.shape)
    
    def updatestate(self,x_input):
        cdef float[::1] layers_memview = self.layers
        cdef float[::1] weights_memview = self.weights
        matrixMultiplication(&weights_memview[0], &layers_memview[0], 1, <float> lam, <float> beta)  ### TODO: Replace with Thomas's Matrix Mutliply
        # input = x_input + int_input
        # d_layers = (-1 * self.layers) + sigmoid(input,self.lam,self.beta)
		
        # self.layers = self.layers + d_layers * self.timestep
        self.output = self.layers


## W = 1's are perfect integrators
## <1 is a leaky integrator
## >1 is a switch


## Specify the desired params here
W = np.ndarray(shape=(2,2), dtype=np.float32)
W = np.ascontiguousarray([2], dtype=np.float32)
cdef np.ndarray V = np.ndarray(shape=(2,2),dtype=np.float32)
V = np.ascontiguousarray([0], dtype=np.float32)



lam = 4;
beta = 0.5;
timestep = .1

# Input changing over time, response follows

t = [0] * 100

ext_input = [0] * 30
ext_input.extend([1]*40)
ext_input.extend([0.5]*30)
output = [0] * 100

hn = HysteresisNetwork(W,V,lam,beta,timestep,ext_input)

for m in range(1,100,1):
    t[m] = m
    x_input = ext_input[m]
    hn.updatestate(x_input)
    output[m] = hn.output
    print(hn.output)

	
print("\n")	
	
# plot output
#plt.ion
'''
plt.figure(0)
plt.plot(t, ext_input)
plt.plot(t, output)
plt.axis([0,100,0,1])
plt.xlabel('Time')
plt.ylabel('Output Level (V)')
plt.title('Time Response Curve')


#######
# Loop over all ext_input for hysteresis plot

itr = 0

ext_input = [0] * 201
output = [0] * 201

for m in range(-100,100,1):

    itr = itr + 1
    ext_input[itr] = float(m) / 100;
    hn = HysteresisNetwork(W,V,lam,beta,timestep,ext_input[itr])
    x_input = ext_input[itr]

    for i in range(1000):
        hn.updatestate(x_input)

    output[itr] = hn.output



# plot output
#plt.ion
plt.figure(1)
plt.plot(ext_input, output)
plt.axis([-1,1,0,1])
plt.xlabel('External Input Level')
plt.ylabel('Output Level (V)')
plt.title('Network Response Curve')

plt.show()
'''

