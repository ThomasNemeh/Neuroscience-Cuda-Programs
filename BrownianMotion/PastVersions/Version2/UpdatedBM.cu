// Program corresponding to CythonBM.cu that can be run directly from the command lin. For testing purposes.
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda.h> 
#include <cmath>
#include <ctime>
#include "book.h"

// Error handling code used in Nvidia example found here: https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-options
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
	
int *failCross = nullptr;

//Function to generate brownian path, which is stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double *results, int *crossTimes, double T, int N, int numSims, double lowerThreshold, double upperThreshold, int *dev_failCross, double seconds) {
	int crossTimeIndex = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (crossTimeIndex < numSims) {
		curandState_t state;
		
		curand_init (blockIdx.x * 1000 + threadIdx.x + seconds, 0, 0, &state);
		double random;
		int start = (threadIdx.x + blockIdx.x * blockDim.x) * N;  
		
		bool crossed = false;
	
		crossTimes[crossTimeIndex] = 0;
		results[start] = 0.0;
	
		for (int j = start + 1; j < start + N; j++) {
			random = curand_normal_double(&state);
			results[j] = results[j-1] + random * sqrt((double) T / N);
			if (results[j] >= upperThreshold && !crossed) {
				crossTimes[crossTimeIndex] = j - start;
				crossed = true;
			}
			else if (results[j] <= lowerThreshold && !crossed) {
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

int *getFailCross() {
	return failCross;
}

double getAverage(int* array, int numSims) {
	double sum = 0;
	int size = numSims;
	int nonZero = 0;
	for(int i = 0; i < size; i++) {
		if (array[i] != 0) {
			sum += abs(array[i]);
			nonZero++;
		}
	}
	
	return sum/nonZero;
}



int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	time_t timer;
	struct tm y2k = {0};
	double seconds;

	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time(&timer);  /* get current time; same as: timer = time(NULL)  */

	seconds = difftime(timer,mktime(&y2k));

	//Arrays to store the brownian path, one for the host and one for the device
	int N = 500;
	double T = 1;
	int numSims = 1;
	int numBlocks = (511 + numSims) / numSims;
	int numThreads = 512;
	double lowerThreshold = -1;
	double upperThreshold = 1;
	double *results = new double[N * numSims];
	double *dev_results;
	
	failCross = new int;
	int *dev_failCross;
	
	int *crossTimes = new int[numSims];
	int *dev_crossTimes;
	
	// Allocate space for results array on device
	cudaMalloc(&dev_results, N * numSims * sizeof(double));
	cudaMalloc(&dev_crossTimes, numSims * sizeof(int));
	cudaMalloc(&dev_failCross, sizeof(dev_failCross));
	
	//Call GPU function
	randomWalk<<<numBlocks, numThreads>>>(dev_results, dev_crossTimes, T, N, numSims, lowerThreshold, upperThreshold, dev_failCross, seconds);
	
	
	//copy results array from device to host
	cudaMemcpy(results, dev_results , N * numSims * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaMemcpy(crossTimes, dev_crossTimes, numSims * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaMemcpy(failCross, dev_failCross, sizeof(dev_failCross), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.1f ms/n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	printf("\n");
	
	// print out path
	for (int i=0; i < (N * numSims); i++) {
		printf("%f ", results[i]);
		if (i == N - 1) {
			printf("\n");
			printf("\n");
		}
	}
	printf("\n");
	printf("\n");
	printf("\n");
	
	/*
	// print out cross times
	for (int i=0; i < numSims; i++) {
		printf("%d ", crossTimes[i]);
	}
	
	printf("\n");
	printf("\n");
	printf("\n");
	
	int x = *getFailCross();
	printf("%d ", x);
	
	printf("\n");
	printf("\n");
	printf("\n");
	printf("Average crossing time: %f ", getAverage(crossTimes, numSims));
	printf("\n");
	int x = *getFailCross();
	printf("Number that failed to cross: %d ", x);
	printf("\n");
	printf("\n");
	*/
	
	//clean up
	cudaFree(dev_results);
	cudaFree(dev_crossTimes);
	cudaFree(dev_failCross);
	
	return 0;

}


