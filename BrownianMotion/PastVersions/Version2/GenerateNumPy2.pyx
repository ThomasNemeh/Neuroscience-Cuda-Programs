import sys
sys.path.insert(0,"/home/psimen/anaconda3/lib/python3.6/site-packages")                                                                                                      

cimport numpy as np
import numpy as np

from libc.stdlib cimport free
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport *
from libc.stdint cimport uintptr_t
import matplotlib.pyplot as plt

np.import_array()

import pickle

cdef extern from "CythonBM.h":
    double *makePath(int T, int N, int numSims, double lowerThreshold, double upperThreshold);
    int *getCrossTimes();
    int *getFailCross();
	
cdef int numPaths
cdef np.ndarray ndarray2

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size
    cdef bool doubleType

    cdef set_data(self, int size, bool doubleType, void* data_ptr):
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
        self.doubleType = doubleType

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length size
        if (self.doubleType) :
            ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.data_ptr)
        else :
            ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, self.data_ptr)
        
        return ndarray
		
    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.data_ptr)

def getPaths(int T, int N, int numSims, double lowerThreshold, double upperThreshold):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """
    cdef double *array
    cdef np.ndarray ndarray
    cdef int size
    cdef bool doubleType = 1
    # Call the C function
    array =  makePath(T, N, numSims, lowerThreshold, upperThreshold)
    global numPaths
    numPaths = numSims
    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(N * numSims, doubleType, <void*> array) 
    ndarray = np.array(array_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*> array_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(array_wrapper)


    return ndarray
	
def getCrossingTimes():
    cdef int *array
    global ndarray2
    cdef int size
    cdef bool doubleType = 0
    # Call the C function
    array =  getCrossTimes()

    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(numPaths, doubleType, <void*> array)
    ndarray2 = np.array(array_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray2.base = <PyObject*> array_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(array_wrapper)


    return ndarray2
	
def getNoCross():
    return np.count_nonzero(ndarray2 == 0)
	
def getNoCross(array):
    return np.count_nonzero(array == 0)
	
def plotHistogram():
    num_bins = ndarray2.max() - ndarray2.min()
    n, bins, patches = plt.hist(ndarray2, num_bins, facecolor='blue', alpha=0.5)
    plt.yticks(np.arange(0, max(5, getNoCross()), 1.0))
    plt.show()
	
def plotHistogram(array):
    num_bins = array.max() - array.min()
    n, bins, patches = plt.hist(array, num_bins, facecolor='blue', alpha=0.5)
    plt.yticks(np.arange(0, max(5, getNoCross(array)), 1.0))
	
def writeToFile(fileName):
    with open(fileName, "wb") as f:
        pickle.dump(ndarray2, f, pickle.HIGHEST_PROTOCOL)
		
def readFromFile(fileName):
    with open("/home/thomasnemeh/CudaPrograms/" + fileName, "rb") as f:
        array = pickle.load(f)
    return array
	
print("Functions: \n")
print("getPaths(int T, int N, int numSims, double lowerThreshold, double upperThreshold): returns list of each time step of each Brownian Motion simulation. \n")
print("getCrossingTimes(): returns list of time that each of the N simulations crossed either threshold. If the number is negative, then the simulation crossed the ") 
print("lower threshold at the absolute. value of that timestep. If the number is positive, then the simulation crossed the upper threshold at that timestep. If the number ")
print("is 0, then the simulation never crossed the threshold. The number at the first index corresponds to the first simulation in the getPaths() ")
print("array, the second index to the second simulation, etc.\n")
print("getNoCross(): return the number of simulations that never crossed either threshold. \n")
print("plotHistogram(): plots a histogram of the array returned by getCrossTimes(). \n")
print("The following functions are to be used if you cannot access a graphical interface for the histogram from the command prompt: \n")
print("witeToFile(fileName): writes the getCrossTimes() array to a file. \n") 
print("readFromFile(fileName): reads the getCrossTimes() array from a file \n")
print("getNoCross(array): input cross times array after reading and writing to file \n")
print("plotHistogram(array): input cross times array after reading and writing to file \n")






  
	
