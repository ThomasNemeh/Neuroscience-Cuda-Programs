import sys
sys.path.insert(0,"/home/psimen/anaconda3/lib/python3.6/site-packages")                                                                                                      

cimport numpy as np
import numpy as np

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
import matplotlib.pyplot as plt

np.import_array()

import pickle

cdef extern from "HysterisisMM.h":
    void fillWeights(float* weight, int dim);
    void fillLayers(float* layers, int dim);
    void matrixMultiplication(float *layers, float *weights, float *external, int dim, int iterations, float timestep, float L, float M);

# number of dimensions of square matrix
cdef int dimensions = -1
# number of iterations of matrix by vector multiplication
cdef int iterNum = -1
# array that stores all our resulting activation vectors
cdef np.ndarray final_result

# Exceptions used for incorrect arguments in the 'hysterisis' function
class ParameterError(Exception):
    pass	

def hysterisis(int iterations, float dt, float L, float B, np.ndarray layers = None, np.ndarray weights = None, np.ndarray external = None, int num_neurons = -1):
    global dimensions, iterNum, final_result

    #get the number of neurons in our network, which will be the a and y dimension of our weight connections matrix
    cdef int dim
    if num_neurons > 0: 
        dim = num_neurons
    else:
        if weights == None: raise ParameterError('Either weights or mun_neurons must be defined to specify the length of layers')
        else:
            dim = <int> weights.shape[0]
			
    #ensure that there are no errors in the parameters
    if weights is not None:
        if weights.ndim > 1: 
            if (weights.shape[0] != dim or weights.shape[1] != dim): raise ParameterError('The dimensions of the weight connections matrix is not correct.')
            weights = weights.flatten('C')
        else: 
            if weights.size != dim * dim:
                raise ParameterError('The size of the weight matrix is not correct')
			
    if layers is not None:
        if (layers.ndim > 1): raise ParameterError('layers should be a flat vector')
        if (layers.size != dim): raise ParameterError('The size of the weight matrix and neurons vector do not match')
		
    total_size = dim * iterations + dim
    if external is not None:
        initial_padding = np.ndarray(shape=(dim,), buffer=np.zeros(dim), dtype=np.float32, order='C')
        external = np.append(initial_padding, external)
        external_size = external.size
        if (external.ndim > 1): raise ParameterError('external inputs should be a flat vector')
        if (external_size > total_size): raise ParameterError('The external inputs vectors is too large (should be <= num_neurons * iterations + num_neurons')
			
    #preprosessing of weights matrix. Fill with random values if not supplied by user.
    cdef float[::1] weights_memview
    layers_size = dim * iterations + dim
    if weights is None:
        weights = np.ndarray(shape=(dim * dim,), dtype=np.float32, order = 'C')
        weights_memview = weights
        fillWeights(&weights_memview[0], dim)
    else:
        weights_memview = weights
        if not weights.flags['C_CONTIGUOUS']: weights = np.ascontiguousarray(weights) # Makes a contiguous copy of the numpy array.
			
    #preprosessing of layers vector. Fill with random values if not supplied by user.
    cdef float[::1] layers_memview    
    if layers is None:
        layers = np.ndarray(shape=(layers_size,), dtype=float, order='C')
        layers_memview = layers
        fillLayers(&layers_memview[0], dim)
    else: 
        if layers.shape[0] != 1: raise ParameterError('Layers must be a flat numpy array')
		#The layers vector will hold the results for each iteration of the operation of our network
        results = np.ndarray(shape=(layers_size - dim,), dtype=np.float32, order='C')
        layers = np.append(layers, results)
        layers_memview = layers
		
    #preprosessing of external inputs vector. Fill with zeros if not supplied by user.
    cdef float[::1] external_memview
    if external is None:
        external = np.ndarray(shape=(layers_size,), buffer=np.zeros(layers_size), dtype=np.float32, order='C')
    else:
        if (external_size < total_size):
            #padd out external inputs vector with zeros if size is less that dim * iterations + dim		
            print('external inputs size is less than total size of simulation. Padding out external with zeros...')
            rest_size = total_size - external_size
            rest_external = np.ndarray(shape=(rest_size,), buffer=np.zeros(rest_size), dtype=np.float32, order='C')
            external = np.append(external, rest_external)
		
    if not external.flags['C_CONTIGUOUS']:
        external = np.ascontiguousarray(external, dtype=np.float32) # Makes a contiguous copy of the numpy array.
		
    external_memview = external
		
    #TODO: add check that dimensions of arrays match
	
    #print("dimensions: " + str(dim))
    #print("layers.shape: " + str(layers.shape[0]))
    #print("layers: " + str(layers))
    #print("weights.shape: " + str(weights.shape[0]))
    #print("weights: " + str(weights))
    #print("external.shape: " + str(external.shape[0]))
    #print("external inputs: " + str(external))
    #print("timestep: " + str(dt))
    #print("lam: " + str(L))
    #print("beta: " + str(B))
	
    matrixMultiplication(&layers_memview[0], &weights_memview[0], &external_memview[0], dim, iterations, dt, L, B)
	
    dimensions = dim
    iterNum = iterations
    final_result = layers
	
    return layers
	

# Exceptions used for incorrect arguments in the 'hysterisis' function
class CallError(Exception):
    pass	
	
# Scatter plot of our list of activation vectors after last instance of running the 'hysterisis' function
def plotResults():
    global final_result, dimensions
    if (dimensions == -1): raise CallError('No network has been processed')
    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
    m, n = 0
    xValues = np.arange(1, dimensions * dimensions + 1)
    while m < iterNum:
        plt.scatter(xValues, final_result, colors[n])
        n = n + 1
        if n > 7:
            n = 0
        m = m + 1
        plt.show
		
# Scatter plot of our list of activation vectors from given array
def plotResults(array, length, numVectors):
    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
    m, n = 0
    xValues = np.arange(1, length * length + 1)
    while m < numVectors:
        plt.scatter(xValues, array, colors[n])
        n = n + 1
        if n > 7:
            n = 0
        m = m + 1
        plt.show

# write list of activation vectors to file
def writeResultToFile(fileName):
    global final_result
    with open(fileName, "wb") as f:
        pickle.dump(final_result, f, pickle.HIGHEST_PROTOCOL)

# print list of activation vectors as sequence of vertical vectors
def printResults():
    global final_result, dimensions
    if (dimensions == -1): raise CallError('No network has been processed')
    cdef int i = 0
    cdef int j
    vector = ""
    while i < iterNum: 
        j = 0
        while j < dimensions:
            vector += str (final_result[i * dimensions + j]) + " " 
            j = j + 1
        i = i + 1
        vector += "\n"
    print(vector)
	
# print list of activation vectors from given parameters
def printResults(array, length, numVectors):
    cdef int i = 0
    cdef int j
    vector = ""
    while i < numVectors: 
        j = 0
        while j < length:
            vector += str (array[i * length + j]) + " " 
            j = j + 1
        i = i + 1
        vector += "\n"
    print(vector)
	
# function to read an array stored in a file
def readFromFile(fileName):
    with open("/home/thomasnemeh/CudaPrograms/" + fileName, "rb") as f:
        array = pickle.load(f)
    return array

# print instructions
def printInstructions():
    print("Functions: \n")
    print("hysterisis(int iterations, float dt, float L, float B, np.ndarray layers = None, np.ndarray weights = None, np.ndarray external = None, int num_neurons = -1): neural network")
    print("with neurons given by layers, weight connections given by the weights matrix, and external inputs at each iteration given by the external vector. If None, weights and layers")
    print("will be filled with random values. If None, external will be filled with zeros.")
    print("This will be multiplied by the matrix again, and this process repeats for the number of iterations entered. L and B are values to be entered in the following squeeze function:") 
    print("f(x) = 1 / (1 + exp(-L * (x - B))). This function will be applied to every element of the resulting vector to keep values bounded between 0 and 1. The result of this function")
    print(" will be a 1-dimensional array containing the results from each iteration.\n")
    print("plotResults(): scatter plot of the points in each vector created in each iteration. Points from different vectors will have different colors. There are only 8 colors available, ")
    print("so if there are more than 8 iterations, there will have to be repeating colors.\n") 
    print("plotResults(array, length, numVectors): scatter plot from given array of activation vectors, given the length of each activation vector and number of activation vectors in the array.\n")
    print("printResults(): prints sequence of activation vectors. The vectors are printed out vertically. \n")
    print("printResults(array, length, numVectors): prints out supplied array of activation vectors, given length of each activation vector and the number of vectors in the array.\n")
    print("writeResultsToFile(fileName): writes array produced by the multiplication function to the given file.\n")
    print("readFromFile(fileName): reads the array stored in the given files. \n")
    print("Note: functions that do not take an array as a parameter rely on the array of activation vectors calculated in the last use 'multiplication' function\n")
