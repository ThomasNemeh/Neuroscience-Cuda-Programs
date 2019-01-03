// A program to generate a 1-dimensional NumPy array that stores the user’s desired number of Brownian Paths, all generated in parallel on crispr’s 3 GPU’s.
// Also generates an array indicating the time at which each path crosses the upper threhsold supplied by the user. See documentation.

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "../book.h"
#include <vector>

int *crossTimes = nullptr;
int *failCross = nullptr;

//Function to generate brownian paths, which are stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double *results, int *crossTimes, int T, int N, int numSims, double upperThreshold, double deviceID) {

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
			results[j] = results[j-1] + random * sqrt((double) T / N);
			
			// store crossing time as positive value if it has crossed the upper threshold. Negative value if crossed the lower threshold
			if (!crossed && results[j] >= upperThreshold) {
				crossTimes[crossTimeIndex] = j - start;
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
	double T;			 // parameter for brownian path equation 
	double upperThreshold;
};

// function to execute on each individual GPU
void* routine(void *voidData) {

	DataStruct *data = (DataStruct*)voidData;
	cudaSetDevice(data->deviceID);
	
	int sims = data->sims;
	
	// allocate arrays on host to store results, as well as temporary arrays on gpu for our global function 
	double *dev_results;
	double *partialResults = (double*)malloc(sims * data->N * sizeof(double));
	int *dev_crossTimes;
	int *partialCrossTimes = (int*)malloc(sims * sizeof(int));
	cudaMalloc(&dev_results, data->N * sims * sizeof(double));
	cudaMalloc(&dev_crossTimes, sims * sizeof(int));
	
	// calculate number of blocks and threads for global function
	int numBlocks = (511 + sims) / sims;
	int numThreads = 512;
	
	// call GPU function
	randomWalk<<<numBlocks, numThreads>>>(dev_results, dev_crossTimes, data->T, data->N, sims, data->upperThreshold, data->deviceID);
	
	// transfer data on gpu to host
	cudaMemcpy(partialResults, dev_results , data->N * sims * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(partialCrossTimes, dev_crossTimes , sims * sizeof(int), cudaMemcpyDeviceToHost);
	
	data->resultArray = partialResults;
	data->crossArray = partialCrossTimes;
	
	// free gpu memory
	cudaFree(dev_results);
	cudaFree(dev_crossTimes);
	
	return 0;
}

// host function to generate the results and crossTimes arrays, and then return the results array
// defined in 3gpubm.h in order to import into cython code (see GenerateNumPy3.pyx)
int main() {
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	double T = 1;
	int N = 6;
	int numSims = 20;
	double upperThreshold = 1;
	
	// fill a data structure of each of crispr's 3 gpu's
	DataStruct data[3];
	data[0].deviceID = 0;
	data[0].sims = numSims / 3;
	data[0].N = N;
	data[0].T = T;
	data[0].upperThreshold = upperThreshold;
	data[1].deviceID = 0;
	data[1].sims = numSims / 3;
	data[1].N = N;
	data[1].T = T;
	data[1].upperThreshold = upperThreshold;
	data[2].deviceID = 0;
	data[2].sims = numSims / 3 + numSims % 3;
	data[2].N = N;
	data[2].T = T;
	data[2].upperThreshold = upperThreshold;
	
	// start a separate thread for each gpu
	CUTThread thread = start_thread(routine, &(data[0]));
	CUTThread thread2 = start_thread(routine, &(data[1]));
	routine(&(data[2]));
	end_thread(thread);
	end_thread(thread2);
	
	double *results = new double[N * numSims];  // the main array to store the path for each simulations, with an index for each point along the path
	crossTimes = new int[numSims];				// the array to store the cross time for each simulation

	// get output of each gpu and concatenate the arrays
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
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//printf("Time to generate: %3.1f ms/n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	for (int i=0; i < (N * numSims); i++) {
		printf("%f ", results[i]);
	}
	printf("\n");
	printf("\n");
	
	for (int i=0; i < (numSims); i++) {
		printf("%d ", crossTimes[i]);
	}
	printf("\n");
	
	
	return 0;

}





