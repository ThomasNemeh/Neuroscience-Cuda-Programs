#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

int *crossTimes = nullptr;
int *failCross = nullptr;

//Function to generate brownian path, which is stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double *results, int *crossTimes, int T, int N, int numSims, double lowerThreshold, double upperThreshold, int *dev_failCross, double seconds) {
	int crossTimeIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (crossTimeIndex < numSims) {
		curandState_t state;
		curand_init (blockIdx.x * 1000 + threadIdx.x + seconds, 0, 0, &state);
		double random;
		int start = (threadIdx.x + blockIdx.x * blockDim.x) * N;  
	
		crossTimes[crossTimeIndex] = 0;
		results[start] = 0.0;
		bool crossed = false;
	
		for (int j = start + 1; j < start + N; j++) {
			random = curand_normal_double(&state);
			results[j] = results[j-1] + random * sqrt((double) T / N);
			if (!crossed && results[j] >= upperThreshold) {
				crossTimes[crossTimeIndex] = j - start;
				crossed = true;
			}
			else if (!crossed && results[j] <= lowerThreshold) {
				crossTimes[crossTimeIndex] = -1 * (j - start);
				crossed = true;
			}
		}
		
		if (!crossed) {
			atomicAdd(dev_failCross, 1);
		}
	
	}
	
	
	/*
	Generate 2 doubles at once. Test later to see if this is more efficient:
	double curand_normal2_double (state);  
	*/
	
}

extern "C++" double *makePath(int T, int N, int numSims, double lowerThreshold, double upperThreshold) {
	//Arrays to store the brownian path, one for the host and one for the device
	double *results = new double[N * numSims];
	double *dev_results;
	
	//values to calculate how many simulations failed to cross either boundry
	failCross = new int;
	int *dev_failCross;
	
	//Calculate number of blocks and threads for global function
	int numBlocks = (511 + numSims) / numSims;
	int numThreads = 512;
	
	//Array to store threshold crossing time for each path
	crossTimes = new int[numSims];
	int *dev_crossTimes;
	
	//Get system time for random number seed
	time_t timer;
	struct tm y2k = {0};
	double seconds;
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;
	time(&timer);  /* get current time; same as: timer = time(NULL)  */
	seconds = difftime(timer,mktime(&y2k));
	
	// Allocate space for arrays on device
	cudaMalloc(&dev_results, N * numSims * sizeof(double));
	cudaMalloc(&dev_crossTimes, numSims * sizeof(int));
	cudaMalloc(&dev_failCross, sizeof(dev_failCross));
	
	//Call GPU function
	randomWalk<<<numBlocks, numThreads>>>(dev_results, dev_crossTimes, T, N, numSims, lowerThreshold, upperThreshold, dev_failCross, seconds);
	
	//copy results array from device to host
	cudaMemcpy(results, dev_results , N * numSims * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(crossTimes, dev_crossTimes, numSims * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(failCross, dev_failCross, sizeof(dev_failCross), cudaMemcpyDeviceToHost);
	
	//clean up
	cudaFree(dev_results);
	cudaFree(dev_crossTimes);
	cudaFree(dev_failCross);
	
	return results;
}

extern "C++" int *getCrossTimes() {
	return crossTimes;
}

extern "C++" int *getFailCross() {
	return failCross;
}


