// A program to generate a 1-dimensional NumPy array that stores the user’s desired number of Brownian Paths, all generated in parallel on crispr’s 3 GPU’s.

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "book.h"	// Code supplied by Nvidia for error handling and handling multiple GPUs- corresponds to Cuda By Example book.
#include <vector>	// use std::copy to concatenate arrays

// array to store crossing time of each simulation, positive time for upper threshold and negative time for lower threshold. 0 if never crossed.
int *crossTimes = nullptr;

//Function to generate brownian paths, which are stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double *results, int *crossTimes, int T, int N, double drift, int numSims, double lowerThreshold, double upperThreshold, double deviceID) {

	// a variable to keep track of this simulation's position in the crossTimes array
	int crossTimeIndex = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (crossTimeIndex < numSims) {
		
		// create random number generator
		curandState_t state;
		curand_init (blockIdx.x * (1000 * deviceID) + threadIdx.x + clock64(), 0, 0, &state);
		double random;
		
		// starting position of this siulation in results array
		int start = (threadIdx.x + blockIdx.x * blockDim.x) * N;  
		
		// set default value of cross time for this simulation to 0, since the simulation hasn't crossed the threshold yet
		crossTimes[crossTimeIndex] = 0;
		
		// starting point of path is 0
		results[start] = 0.0;
		
		// boolean to keep track of whether this path has crossed
		bool crossed = false;
	
		for (int j = start + 1; j < start + N; j++) {
			// generate random number
			random = curand_normal_double(&state);
			
			//calculate next step of path
			results[j] = results[j-1] + drift * ((double) T / N) + random * sqrt((double) T / N);
			
			// store crossing time as positive value if it has crossed the upper threshold. Negative value if crossed the lower threshold
			if (!crossed && results[j] >= upperThreshold) {
				crossTimes[crossTimeIndex] = j - start;
				crossed = true;
			}
			else if (!crossed && results[j] <= lowerThreshold) {
				crossTimes[crossTimeIndex] = -1 * (j - start);
				crossed = true;
			}
		}
	
	}
	
}

// data structure to hold information for each GPU
struct DataStruct {
	int deviceID; 		 // id of gpu
	int sims;			 // number of simulations to be executed on this gpu
	double *resultArray; // array to store brownian paths calculated on this gpu
	int *crossArray;	 // array to store cross times calculates on this gpu
	int N;				 // number of simulations on this gpu
	double T;	       	 // parameter for brownian path equation 
	double drift;		 // drift parameter brownian path equation
	double lowerThreshold;
	double upperThreshold;
};

// function to execute on each individual GPU
void* routine(void *voidData) {

	DataStruct *data = (DataStruct*)voidData;
	HANDLE_ERROR(cudaSetDevice(data->deviceID));
	
	int sims = data->sims;
	
	// allocate arrays on host to store results, as well as temporary arrays on gpu for our global function 
	double *dev_results;
	double *partialResults = (double*)malloc(sims * data->N * sizeof(double));
	int *dev_crossTimes;
	int *partialCrossTimes = (int*)malloc(sims * sizeof(int));
	HANDLE_ERROR(cudaMalloc(&dev_results, data->N * sims * sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&dev_crossTimes, sims * sizeof(int)));
	
	// calculate number of blocks and threads for global function
	int numBlocks = (511 + sims) / sims;
	int numThreads = 512;
	
	// call GPU function
	randomWalk<<<numBlocks, numThreads>>>(dev_results, dev_crossTimes, data->T, data->N, data->drift, sims, data->lowerThreshold, data->upperThreshold, data->deviceID);
	
	// transfer data on gpu to host
	HANDLE_ERROR(cudaMemcpy(partialResults, dev_results , data->N * sims * sizeof(double), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(partialCrossTimes, dev_crossTimes , sims * sizeof(int), cudaMemcpyDeviceToHost));
	
	data->resultArray = partialResults;
	data->crossArray = partialCrossTimes;
	
	// free gpu memory
	HANDLE_ERROR(cudaFree(dev_results));
	HANDLE_ERROR(cudaFree(dev_crossTimes));
	
	return 0;
}

// host function to generate the results and crossTimes arrays, and then return the results array
// defined in 3gpubm.h in order to import into cython code (see GenerateNumPy3.pyx)
extern "C++" double *makePath(double T, int N, double drift, int numSims, double lowerThreshold, double upperThreshold) {
	
	// fill a data structure of each of crispr's 3 gpu's
	DataStruct data[3];
	data[0].deviceID = 0;
	data[0].sims = numSims / 3 + numSims % 3;
	data[0].N = N;
	data[0].T = T;
	data[0].drift = drift;
	data[0].lowerThreshold = lowerThreshold;
	data[0].upperThreshold = upperThreshold;
	if (numSims > 2) {
		data[1].deviceID = 0;
		data[1].sims = numSims / 3;
		data[1].N = N;
		data[1].T = T;
		data[1].drift = drift;
		data[1].lowerThreshold = lowerThreshold;
		data[1].upperThreshold = upperThreshold;
		data[2].deviceID = 0;
		data[2].sims = numSims / 3;
		data[2].N = N;
		data[2].T = T;
		data[2].drift = drift;
		data[2].lowerThreshold = lowerThreshold;
		data[2].upperThreshold = upperThreshold;
	
		// start a separate thread for each gpu
		CUTThread thread = start_thread(routine, &(data[0]));
		CUTThread thread2 = start_thread(routine, &(data[1]));
		routine(&(data[2]));
		// wait until threads have finished before continuing 
		end_thread(thread);
		end_thread(thread2);
	}
	else {	
		routine(&(data[0]));	// only one gpu is necessary if there are less than 3 simulations
	}
	
	double *results = new double[N * numSims];  // the main array to store the path for each simulations, with an index for each point along the path
	crossTimes = new int[numSims];				// the array to store the cross time for each simulation
	
	
	// get output of each gpu and concatenate the arrays
	if (numSims > 2) {
		double *arr1 = data[0].resultArray;
		int size1 = data[0].sims * N;
		double *arr2 = data[1].resultArray;
		int size2 = data[1].sims * N;
		double *arr3 = data[2].resultArray;
		int size3 = data[2].sims * N;
	
		std::copy(arr1, arr1 + size1, results);
		std::copy(arr2, arr2 + size2, results + size1);
		std::copy(arr3, arr3 + size3, results + size1 + size2);
		
		int *carr1 = data[0].crossArray;
		size1 = data[0].sims;
		int *carr2 = data[1].crossArray;
		size2 = data[1].sims;
		int *carr3 = data[2].crossArray;
		size3 = data[2].sims;
	
		std::copy(carr1, carr1 + size1, crossTimes);
		std::copy(carr2, carr2 + size2, crossTimes + size1);
		std::copy(carr3, carr3 + size3, crossTimes + size1 + size2);
	}
	else {	// only one gpu is used if there are less than 3 simulations
		results = data[0].resultArray;
		crossTimes = data[0].crossArray;
	}
	
	return results;
}

// return crossTies array
// defined in 3gpubm.h in order to import into cython code (see GenerateNumPy3.pyx)
extern "C++" int *getCrossTimes() {
	return crossTimes;
}

