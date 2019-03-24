import sys
sys.path.insert(0,"/home/psimen/anaconda3/lib/python3.6/site-packages")

cimport numpy as np
import numpy as np

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
import matplotlib.pyplot as plt

np.import_array()

import pickle

from copy import copy

cdef extern from "HysterisisMM.h":
    void fillWeights(double* weight, int dim);
    void fillLayers(double* layers, int dim);
    void matrixMultiplication(double *layers, double *weights, double *external, double *lamBeta, int dim, int iterations, float timestep, double noise);

# number of dimensions of square matrix
cdef int dimensions = -1
# number of iterations of matrix by vector multiplication
cdef int iterNum = -1
# array that stores all our resulting activation vectors
cdef np.ndarray final_result = None
# reshaped array for plotting purposes
cdef np.ndarray multi_d_result = None

# Exceptions used for incorrect arguments in the 'hysterisis' function
class ParameterError(Exception):
    pass

def hysterisis(int iterations, float dt, double noise, lam_beta, np.ndarray layers = None, np.ndarray weights = None, np.ndarray external = None, int num_neurons = -1):

    #get the number of neurons in our network, which will be the a and y dimension of our weight connections matrix
    cdef int dim
    if num_neurons > 0:
        dim = num_neurons
    else:
        if weights is None and layers is None: raise ParameterError('num_neurons must be defined if weights and layers is not provided')
        else:
            if weights is not None: dim = <int> weights.shape[0]
            else: dim = <int> layers.shape[0]

    if type(lam_beta) is tuple:
        if len(lam_beta) != 2: raise ParameterError('lam_beta parameter should be of the form (lam, beta), or a list of tuples')
        squeeze = np.tile(np.asarray(lam_beta), dim)
        squeeze = np.ascontiguousarray(squeeze) # Makes a contiguous copy of the numpy array.
    elif type(lam_beta) is list:
        for tup in lam_beta:
            if type(tup) is not tuple or len(tup) != 2: raise ParameterError('each element of list should be of the form (lam, beta)')
        if len(lam_beta) < dim: raise ParameterError('there must be a (lam, beta) pair for each neuron')
        lam_beta = list(sum(lam_beta, ()))
        squeeze = np.ascontiguousarray(lam_beta, dtype=np.float64)
    elif type(lam_beta) is np.ndarray:
        for tup in lam_beta:
            if type(tup) is not np.ndarray or tup.size != 2: raise ParameterError('each element of list should be of the form (lam, beta)')
        if len(lam_beta) < dim: raise ParameterError('there must be a (lam, beta) pair for each neuron')
        lam_beta = lam_beta.flatten()
        squeeze = np.ascontiguousarray(lam_beta, dtype=np.float64)

    cdef double[::1] lam_beta_memview = squeeze

    #ensure that there are no errors in the parameters
    if weights is not None:
        if weights.ndim > 1:
            if (weights.shape[0] != dim or weights.shape[1] != dim): raise ParameterError('The dimensions of the weight connections matrix is not correct.')
            if (weights.shape[0] > 1): weights = weights.flatten()
        else:
            if weights.size != dim * dim:
                raise ParameterError('The size of the weight matrix is not correct')

    if layers is not None:
        if (layers.ndim > 1): raise ParameterError('layers should be a flat vector')
        if (layers.size != dim): raise ParameterError('The size of the weight matrix and neurons vector do not match')

    if external is not None:
        if (external.ndim != 2): raise ParameterError('external input should be a 2D array')
        if (len(external) != dim): raise ParameterError('there should be one external inputs array for each neuron')

    #preprosessing of weights matrix. Fill with random values if not supplied by user.
    cdef double[::1] weights_memview
    if weights is None:
        weights = np.ndarray(shape=(dim * dim,), dtype=np.float64, order = 'C')
        weights_memview = weights
        #fillWeights(&weights_memview[0], dim)
    else:
        if weights.dtype != np.float64: weights = weights.astype(np.float64)
        if not weights.flags['C_CONTIGUOUS']: weights = np.ascontiguousarray(weights) # Makes a contiguous copy of the numpy array.
        weights_memview = weights

    #preprosessing of layers vector. Fill with random values if not supplied by user.
    cdef double[::1] layers_memview
    layers_size = dim * iterations + dim
    if layers is None:
        layers = np.ndarray(shape=(layers_size,), dtype=np.float64, order='C')
        layers_memview = layers
        #fillLayers(&layers_memview[0], dim)
    else:
		#The layers vector will hold the results for each iteration of the operation of our network
        results = np.ndarray(shape=(layers_size - dim,), dtype=np.float64, order='C')
        if layers.dtype != np.float64: layers = layers.astype(np.float64)
        if not layers.flags['C_CONTIGUOUS']: layers = np.ascontiguousarray(layers) # Makes a contiguous copy of the numpy array.
        layers = np.append(layers, results)

        layers_memview = layers

    #preprosessing of external inputs vector. Fill with zeros if not supplied by user.
    cdef double[::1] external_memview
    external_size = layers_size - dim
    if external is None:
        external_flat = np.ndarray(shape=(external_size,), buffer=np.zeros(external_size), dtype=np.float64, order='C')
    else:
        #external_flat = np.empty((external_size,), dtype=np.float64, order='C')
        external_flat = np.ascontiguousarray([], dtype = np.float64)
        ext_count = 0
        print("external_size: " + str(external_size))
        print("len(external): " + str(len(external)))
        while ext_count < len(external):
            if external[ext_count].size < iterations:
                rest_size =  iterations - external[ext_count].size
                rest_external = np.ndarray(shape=(rest_size,), buffer=np.zeros(rest_size), dtype=np.float64, order='C')
                arr_ext = copy(external[ext_count])
                arr_ext = np.append(arr_ext, rest_external)
                external_flat = np.append(external_flat, arr_ext)
                print("arr_ext_" + str(ext_count) + " size: " + str(arr_ext.size))
                print("external_flat " + str(ext_count) + " size: " + str(external_flat.size))
                ext_count += 1

    external_memview = external_flat

    '''
    print("dimensions: " + str(dim))
    print("iterations: " + str(iterations))
    print("layers.shape: " + str(layers.shape[0]))
    print("layers: " + str(layers))
    print("weights.shape: " + str(weights.shape[0]))
    '''
    #print("weights: " + str(weights))

    print("external.shape: " + str(external_flat.shape[0]))
    print("external inputs: " + str(external_flat))
    '''
    print("timestep: " + str(dt))
    print("lam: " + str(L))
    print("beta: " + str(B))
    '''

    matrixMultiplication(&layers_memview[0], &weights_memview[0], &external_memview[0], &lam_beta_memview[0], dim, iterations, dt, noise)

    global dimensions, iterNum, final_result
    dimensions = dim
    iterNum = iterations
    final_result = layers

    return layers


# Exceptions used for incorrect arguments in the 'hysterisis' function
class CallError(Exception):
    pass

def plotLastResult():
    global final_result, dimensions, multi_d_result

    if (dimensions == -1): raise CallError('No network has been processed')
    if multi_d_result is None:
        multi_d_result = final_result.reshape(dimensions, iterNum + 1)

    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])

    for layer in range(dimensions):
        plt.plot(multi_d_result[layer,:])

def plotNetwork(network, dimensions, iterations):
    if dimensions == -1:
        raise CallError('No network has been processed')

    multi_d_result = network.reshape(dimensions, iterations + 1)

    #different colors are used automatically?
    #colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])

    for layer in range(dimensions):
          plt.plot(multi_d_result[layer,:])

def plotNeuronsLastResult(neurons):
    global dimensions, multi_d_result, iterNum

    if dimensions == -1:
        raise CallError('No network has been processed')

    if multi_d_result is None:
        multi_d_result = final_result.reshape(dimensions, iterNum + 1)

    for layer in neurons:
        if layer >= dimensions:
            raise ParameterError('neuron number must be < dim (neurons indexed from 0)')
        plt.plot(multi_d_result[layer,:])

def plotNeuons(network, dimensions, iterations, neurons):

    multi_d_result = network.reshape(dimensions, iterations + 1)

    for layer in neurons:
        if layer >= dimensions:
            raise ParameterError('neuron number must be < dim (neurons indexed from 0)')
        plt.plot(multi_d_result[layer,:])

# write list of activation vectors to file
def writeResultToFile(fileName):
    global final_result
    with open(fileName, "wb") as f:
        pickle.dump(final_result, f, pickle.HIGHEST_PROTOCOL)

# write list of activation vectors to file
def writeResultToFile(fileName, array):
    with open(fileName, "wb") as f:
        pickle.dump(array, f, pickle.HIGHEST_PROTOCOL)

# print list of activation vectors as sequence of vertical vectors
def printLastResults():
    global final_result, dimensions
    vector = ""
    if (dimensions == -1): raise CallError('No network has been processed')

    for m in range(1,dimensions + 1,1):
        vector += "Neuron " + str(m) + "  "
    vector+="\n"

    cdef int i = 0
    cdef int j
    while i < iterNum + 1:
        j = 0
        vector += str(i) + ": "
        while j < dimensions:
            vector += str (final_result[i * dimensions + j]) + " "
            j = j + 1
        i = i + 1
        vector += "\n"
    print(vector)

# print list of activation vectors from given parameters
def printResults(array, length, numVectors):
    vector = ""

    for m in range(1,dimensions + 1,1):
        vector += "Neuron " + str(m) + "  "
    vector+="\n"

    cdef int i = 0
    cdef int j
    while i < numVectors + 1:
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
    print("hysterisis(int iterations, float dt, float noise, float L, float B, np.ndarray layers = None, np.ndarray weights = None, np.ndarray external = None, int num_neurons = -1): neural network")
    print("with neurons given by layers, weight connections given by the weights matrix, and external inputs at each iteration given by the external vector. If None, weights and layers")
    print("will be filled with random values. If None, external will be filled with zeros.")
    print("This will be multiplied by the matrix again, and this process repeats for the number of iterations entered. L and B are values to be entered in the following squeeze function:")
    print("f(x) = 1 / (1 + exp(-L * (x - B))). This function will be applied to every element of the resulting vector to keep values bounded between 0 and 1. The result of this function")
    print(" will be a 1-dimensional array containing the results from each iteration.\n")

    '''print("plotLastResults(): scatter plot of the points in each vector created in each iteration. Points from different vectors will have different colors. There are only 8 colors available, ")
    print("so if there are more than 8 iterations, there will have to be repeating colors.\n")

    print("plotResults(array, length, iterations): scatter plot from given array of activation vectors, given the length of each activation vector and number of activation vectors in the array.\n")'''

    print("printLastResults(): prints sequence of activation vectors. The vectors are printed out vertically. \n")

    print("printResults(array, length, numVectors): prints out supplied array of activation vectors, given length of each activation vector and the number of vectors in the array.\n")

    print("writeResultsToFile(fileName): writes array produced by the hysterisis function to the given file.\n")

    print("readFromFile(fileName): reads the array stored in the given files. \n")

    print("Note: functions that do not take an array as a parameter rely on the array of activation vectors calculated in the last use 'hysterisis' function\n")
