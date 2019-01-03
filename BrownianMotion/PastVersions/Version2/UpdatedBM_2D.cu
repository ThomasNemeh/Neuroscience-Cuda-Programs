// Program corresponding to CythonBM.cu that can be run directly from the command line. For testing purposes.
// Attempt to use 2D array. Doesn't work.

//#include <cmath>
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda.h>

// Error handling code used in Nvidia example found here: https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-options
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

//Function to generate brownian path, which is stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double **results, int T, int N, int numSims) {
	curandState_t state;
	curand_init (1234, 0, 0, &state);
	double random;
	int start = (threadIdx.x + blockIdx.x * blockDim.x);  
	
	if (start < numSims) {
		results[start][0] = 0.0;
	
		for (int j = 1; j < N; j++) {
			random = curand_normal_double(&state);
			results[start][j] = results[start][j-1] + random * sqrt((double) T / N);
		}
	
		/*
		Generate 2 doubles at once. Test later to see if this is more efficient:
		double curand_normal2_double (state);
		*/
	
	}
	
}

int main() {
	//Arrays to store the brownian path, one for the host and one for the device
	const int N = 10;
	int T = 1;
	const int numSims = 5;
	int numBlocks = (127 + numSims) / numSims;
	int numThreads = 128;
	double** results = new double*[numSims];
	for(int i = 0; i < numSims; ++i)
		results[i] = new double[N];
		
	double** dev_results = new double*[numSims];
	for(int i = 0; i < numSims; ++i)
		dev_results[i] = new double[N];
	
	// Allocate space for results array on device
	CUDA_CALL(cudaMalloc(&dev_results, N * numSims * sizeof(double)));
	
	//Call GPU function, with ony one block and one thread
	randomWalk<<<numBlocks, numThreads>>>(dev_results, T, N, numSims);
	
	//copy results array from device to host
	CUDA_CALL(cudaMemcpy(results, dev_results , N * numSims * sizeof(double), cudaMemcpyDeviceToHost));
	
	// print out path
	for (int i=0; i< numSims; i++) {
		for (int j = 0; j < N; j++) {
			printf("%f ", results[i][j]);
		}
		printf("\n");
		printf("\n");
	}
	
	
	//clean up
	CUDA_CALL(cudaFree(dev_results));
	
	return 0;

}

