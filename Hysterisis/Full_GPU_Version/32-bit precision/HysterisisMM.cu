//Program to multiply square matrix with activation vector, both filled with random numbers, and then to multiply the resulting vector by the matrix again- repeat for the
//specified number of iterations.

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <iostream>
using std::cout;
using std::endl;
using std::copy;

using namespace std;

// fills matrix with random float
// Param: pointer to matrix, number of rows, number of columns
void GPU_fill_rand(float *matrix, int rows, int cols) {
     // Create a pseudo-random number generator
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

     // Set the seed for the random number generator using the system clock
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

     // Fill the array with uniformly distributed random numbers on the device between 0 and 1, where 0 is included and 1 is excluded
      curandGenerateUniform(prng, matrix, rows * cols);
}

// converts float values to integer between 0 and 10, where 0 is included and 10 is excluded
// Param: pointer to matrix, number of elements in matrix
__global__ void changeValues(float *matrix, int size) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < size) {
		float a = matrix[index] * 10;
		int b = (int) a;
		matrix[index] = (float) b;

	}
}

//Print matrix storage in column-major format
//Param: pointer to matrix, number of rows, number of columns
void print_matrix(float *matrix, int rows, int cols) {

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            //std::cout << matrix[j * rows + i] << " ";
			matrix[j * rows + i] = 0;
        }
        //std::cout << std::endl;
	}
    //std::cout << std::endl;
}

// perform the sqeeze function on each element of the vector resulting from the later iteration of matrix multiplication
// Param: B = pointer to activation vector, dim = starting point of the vector results of the last iteration of matrix multiplication,
// L and M are parameters of the squeeze function
__global__ void updateState(float *B, float *external, int dim, float timestep, float noise, int length, int totalIterations, int iterationNum, float L, float M) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
	if (index >= length && index < length + dim) {
		int neuronNum = index % dim;
		float input = B[index] + external[neuronNum * (totalIterations) + iterationNum];
		float old_output = B[index - dim];
		float d_layers = (-1 * old_output) + 1 / (1 + expf(-1 * L * (input - M)));

		// create random number generator
		curandState_t state;
		curand_init (blockIdx.x * 1000 + threadIdx.x + clock64(), 0, 0, &state);
		float random = curand_normal(&state);
		float guassian_noise = noise * random * sqrt(timestep);
		B[index] = old_output + d_layers * timestep + guassian_noise;
	}
}

// perform the matrix multiplication operation
// Param: handle = handle to the cuBLAS library context. iterations = number of times we multiply activation vector by matrix
//        A = matrix. B = array of activation vectors calculated so far. dim = length & width of square matrix. L, M = parameter for squeeze function
void gpu_blas_mmul(cublasHandle_t &handle, int iterations, float timestep, const float noise, const float *A, float *B, float *external, const int dim, const int size_layers, float L, float M) {
    const float alf = 1; // scalar used for multiplication
    const float bet = 0; // scalar used for multiplication
    const float *alpha = &alf;
    const float *beta = &bet;
	int length = dim;

	for (int i = 0; i < iterations; i++) {
		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, 1, dim, alpha, A, dim, (B + length - dim), dim, beta, (B + length), dim);
		updateState<<<(31 + dim) / 32, 32>>>(B, external, dim, timestep, noise, length, iterations, i, L, M);
		length += dim;
	}
}

extern "C++" void fillWeights(float *weights, int dim) {
	// allocate weight connections matrix on the GPU
	float *dev_weights;
	int size_weights = dim * dim;
	cudaMalloc(&dev_weights, size_weights * sizeof(float));

	// fill matrix and first activation vector with random values
	GPU_fill_rand(dev_weights, dim, dim);

	// change decimal values in matrix to integers between 0 and 10
	changeValues<<<(31 + size_weights) / 32, 32>>>(dev_weights, size_weights);

	// copy results to host
	cudaMemcpy(weights, dev_weights, size_weights * sizeof(float), cudaMemcpyDeviceToHost);

	//Free GPU memory
    cudaFree(dev_weights);
}

extern "C++" void fillLayers(float *layers, int dim) {
	// allocate neurons vector on the GPU
	float *dev_layers;
	int size_layers = dim;
	cudaMalloc(&dev_layers, size_layers * sizeof(float));

	// fill vector vector with random values
	GPU_fill_rand(dev_layers, dim, 1);

	// copy results to host
	cudaMemcpy(layers, dev_layers, dim * sizeof(float), cudaMemcpyDeviceToHost);

	//Free GPU memory
    cudaFree(dev_layers);
}

// external function defined in MatrixMultiplication.h
extern "C++" void matrixMultiplication(float *layers, float *weights, float *external, int dim, int iterations, float timestep, float noise, float L, float M) {
	int size_weights = dim * dim;
	int size_layers = dim * iterations + dim;
	int size_external = size_layers * dim - dim;

	// allocate arrays on device
	float *dev_layers, *dev_weights, *dev_external;
	cudaMalloc(&dev_layers, size_layers * sizeof(float));
	cudaMalloc(&dev_external, size_external * sizeof(float));
	cudaMalloc(&dev_weights, size_weights * sizeof(float));

	// copy arrays to GPU
	cudaMemcpy(dev_layers, layers, size_layers * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_external, external, size_external * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weights, weights, size_weights * sizeof(float),cudaMemcpyHostToDevice);

	// create handle to the cuBLAS library context
	cublasHandle_t handle;
    cublasCreate(&handle);

	gpu_blas_mmul(handle, iterations, timestep, noise, dev_weights, dev_layers, dev_external, dim, size_layers, L, M);

	// destroy handle
	cublasDestroy(handle);

	// copy results to host
	cudaMemcpy(layers, dev_layers, size_layers * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(external, dev_external, size_layers * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(weights, dev_weights, size_weights * sizeof(float),cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(dev_layers);
    cudaFree(dev_weights);
	cudaFree(dev_external);

}
