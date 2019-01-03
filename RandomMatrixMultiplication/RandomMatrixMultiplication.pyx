import sys
sys.path.insert(0,"/home/psimen/anaconda3/lib/python3.6/site-packages")                                                                                                      

cimport numpy as np
import numpy as np

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
import matplotlib.pyplot as plt

np.import_array()

import pickle

cdef extern from "RandomMatrixMultiplication.h":
    float *matrixMultiplication(int dim, int iterations, float L, float M);
    float *getMatrix();

# number of dimensions of square matrix
cdef int dimensions
# number of iterations of matrix by vector multiplication
cdef int iterNum
# array that stores all our resulting activation vectors
cdef np.ndarray ndarray

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        """ Set the data of the array
        This cannot be done in the constructor as it must recieve C-level
        arguments.
        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data            
        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length size
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT, self.data_ptr)
        
        return ndarray
		
    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.data_ptr)

# Python binding of the 'matrixMultiplication' function in 'MatrixMultiplication.h' that does
# not copy the data allocated in C.
def multiplication(int dim, int iterations, float L, float B):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """
    cdef float *array
    global ndarray
    # Call the C function
    array =  matrixMultiplication(dim, iterations, L, B)
    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(dim * iterations, <void*> array) 
    ndarray = np.array(array_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*> array_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(array_wrapper)
	
    global dimensions
    global iterNum
    dimensions = dim
    iterNum = iterations

    return ndarray

# Scatter plot of our list of activation vectors after last instance of running the 'multiplication' function
def plotResults():
    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
    global ndarray
    m, n = 0
    xValues = np.arange(1, dimensions * dimensions + 1)
    while m < iterNum:
        plt.scatter(xValues, ndarray, colors[n])
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
    with open(fileName, "wb") as f:
        pickle.dump(ndarray, f, pickle.HIGHEST_PROTOCOL)

# print list of activation vectors as sequence of vertical vectors
def printResults():
    cdef int i = 0
    cdef int j
    vector = ""
    while i < iterNum: 
        j = 0
        while j < dimensions:
            vector += str (ndarray[i * dimensions + j]) + " " 
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
print("Functions: \n")
print("multiplication(int dim, int iterations, float L, float B): matrix of dimensions dim x dim will be multiplied by activation vector, producing another vector.")
print("This will be multiplied by the matrix again, and this process repeats for the number of iterations entered. L and B are values to be entered in the following squeeze function:") 
print("f(x) = 1 / (1 + exp(-L * (x - B))). This function will be applied to every element of the resulting vector to keep values bounded between 0 and 1. The result of this function")
print(" will be a 1-dimensional array containing the results from each iteration.\n")
print("plotResults(): scatter plot of the points in each vector created in each iteration. Points from different vectors will have different colors. There are only 8 colors available, ")
print("so if there are more than 8 iterations, there will have to be repeating colors.\n") 
print("plotResults(array, length, numVectors): plot from given array of activation vectors, given the length of each activation vector and number of activation vectors in the array.\n")
print("printResults(): prints sequence of activation vectors. The vectors are printed out vertically. \n")
print("printResults(array, length, numVectors): prints out supplied array of activation vectors, given length of each activation vector and the number of vectors in the array.\n")
print("writeResultsToFile(fileName): writes array produced by the multiplication function to the given file.\n")
print("readFromFile(fileName): reads the array stored in the given files. \n")
print("Note: functions that do not take an array as a parameter rely on the array of activation vectors calculated in the last use 'multiplication' function\n")
