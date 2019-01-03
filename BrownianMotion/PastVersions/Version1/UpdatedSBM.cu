// Program corresponding to CythonSBM.cu that can be run directly from the command lin. For testing purposes.

//#include <cmath>
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda.h>

// Error handling code used in Nvidia example found here: https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-options
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

//Function to generate brownian path, which is stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double *results, int T, int N) {
	curandState_t state;
	curand_init (1234, 0, 0, &state);
	double random;
	
	results[0] = 0.0;
	
	for (int j = 1; j < N; j++) {
		random = curand_normal_double(&state);
		results[j] = results[j-1] + random * sqrt((double) T / N);
	}
	
	/*
	Generate 2 doubles at once. Test later to see if this is more efficient:
	double curand_normal2_double (state);
	*/
	
}

int main() {
	//Arrays to store the brownian path, one for the host and one for the device
	int N = 500;
	int T = 1;
	double results[N];
	double *dev_results;
	
	// Allocate space for results array on device
	CUDA_CALL(cudaMalloc(&dev_results, N * sizeof(double)));
	
	//Call GPU function, with ony one block and one thread
	randomWalk<<<1, 1>>>(dev_results, T, N);
	
	//copy results array from device to host
	CUDA_CALL(cudaMemcpy(results, dev_results , N * sizeof(double), cudaMemcpyDeviceToHost));
	
	// print out path
	for (int i=0; i<N; i++) {
		printf("%f ", results[i]);
	}
	printf("\n");
	
	//clean up
	CUDA_CALL(cudaFree(dev_results));
	
	return 0;

}

