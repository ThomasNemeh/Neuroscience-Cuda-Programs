// Program corresponding to CythonBM.cu that can be run directly from the command lin. For testing purposes.
//Attempt to Parallelize function for crossing time. Slower than other methods. 

//#include <cmath>
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda.h>

// Error handling code used in Nvidia example found here: https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-options
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

//Function to generate brownian path, which is stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double *results, int *crossTimes, int T, int N, int numSims) {
	int crossTimeIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (crossTimeIndex < numSims) {
		curandState_t state;
		curand_init (1234, 0, 0, &state);
		double random;
		int start = (threadIdx.x + blockIdx.x * blockDim.x) * N;  
	
		crossTimes[crossTimeIndex] = 0;
		results[start] = 0.0;
	
		for (int j = start + 1; j < start + N; j++) {
			random = curand_normal_double(&state);
			results[j] = results[j-1] + random * sqrt((double) T / N);
		}
	}
	
	/*
	Generate 2 doubles at once. Test later to see if this is more efficient:
	double curand_normal2_double (state);
	*/
	
}

__global__ void getCrossingTimes(double *results, int *crossTimes, int N, int numSims, int lowerThreshold, int upperThreshold) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N * numSims) {
		if (crossTimes[tid/N] == 0) {
			if (results[tid] <= lowerThreshold) {
				crossTimes[tid/N] = tid % N;
			}
			else if (results[tid] >= upperThreshold) {
				crossTimes[tid/N] = tid % N;
			}
		}
		tid += blockDim.x + gridDim.x;
	}
}

int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	//Arrays to store the brownian path, one for the host and one for the device
	int N = 99000;
	int T = 1;
	int numSims = 100000;
	int numBlocks = (127 + numSims) / numSims;
	int numThreads = 128;
	double lowerThreshold = -1;
	double upperThreshold = 1;
	double *results = new double[N * numSims];
	double *dev_results;
	
	int *crossTimes = new int[numSims];
	int *dev_crossTimes;
	
	int numBlocks2 = (511 + N * numSims) / 512;
	
	// Allocate space for results array on device
	CUDA_CALL(cudaMalloc(&dev_results, N * numSims * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_crossTimes, numSims * sizeof(int)));
	
	//Call GPU function, with ony one block and one thread
	randomWalk<<<numBlocks, numThreads>>>(dev_results, dev_crossTimes, T, N, numSims);
	
	//copy results array from device to host
	CUDA_CALL(cudaMemcpy(results, dev_results , N * numSims * sizeof(double), cudaMemcpyDeviceToHost));
	
	getCrossingTimes<<<numBlocks2,512>>>(dev_results, dev_crossTimes, N, numSims, lowerThreshold, upperThreshold);
	
	CUDA_CALL(cudaMemcpy(crossTimes, dev_crossTimes, numSims * sizeof(int), cudaMemcpyDeviceToHost));
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.1f ms/n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	printf("\n");
	
	/*
	// print out path
	for (int i=0; i < (N * numSims); i++) {
		printf("%f ", results[i]);
	}
	printf("\n");
	printf("\n");
	printf("\n");
	// print out cross times
	for (int i=0; i < numSims; i++) {
		printf("%d ", crossTimes[i]);
	}
	printf("\n");
	
	*/
	//clean up
	CUDA_CALL(cudaFree(dev_results));
	
	return 0;

}

