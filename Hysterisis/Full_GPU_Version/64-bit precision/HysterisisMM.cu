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
void GPU_fill_rand(double *matrix, int rows, int cols) {
     // Create a pseudo-random number generator
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

     // Set the seed for the random number generator using the system clock
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

     // Fill the array with uniformly distributed random numbers on the device between 0 and 1, where 0 is included and 1 is excluded
      curandGenerateUniformDouble(prng, matrix, rows * cols);
}

// converts float values to integer between 0 and 10, where 0 is included and 10 is excluded
// Param: pointer to matrix, number of elements in matrix
__global__ void changeValues(double *matrix, int size) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < size) {
		double a = matrix[index] * 10;
		int b = (int) a;
		matrix[index] = (double) b;

	}
}

//Print matrix storage in column-major format
//Param: pointer to matrix, number of rows, number of columns
void print_matrix(double *matrix, int rows, int cols) {

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
__global__ void updateState(double *B, double *external, double *lamBeta, int dim, float timestep, double noise, int length, int totalIterations, int iterationNum) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
	if (index >= length && index < length + dim) {
    int neuronNum = index % dim;
    double lam = lamBeta[neuronNum * 2];
    double beta = lamBeta[neuronNum * 2 + 1];

		double input = B[index] + external[neuronNum * (totalIterations) + iterationNum];
		double old_output = B[index - dim];
		double d_layers = (-1 * old_output) + 1 / (1 + expf(-1 * lam * (input - beta)));

		// create random number generator
		curandState_t state;
		curand_init (blockIdx.x * 1000 + threadIdx.x + clock64(), 0, 0, &state);
		float random = curand_normal(&state);
		double guassian_noise = noise * random * sqrt(timestep);
		B[index] = old_output + d_layers * timestep + guassian_noise;
	}
}

// perform the matrix multiplication operation
// Param: handle = handle to the cuBLAS library context. iterations = number of times we multiply activation vector by matrix
// A = matrix. B = array of activation vectors calculated so far. dim = length & width of square matrix. L, M = parameter for squeeze function
void gpu_blas_mmul(cublasHandle_t &handle, int iterations, float timestep, const double noise, const double *A, double *B, double *external, double *lamBeta, const int dim, const int size_layers) {
    const double alf = 1; // scalar used for multiplication
    const double bet = 0; // scalar used for multiplication
    const double *alpha = &alf;
    const double *beta = &bet;
	  int length = dim;

	for (int i = 0; i < iterations; i++) {
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, 1, dim, alpha, A, dim, (B + length - dim), dim, beta, (B + length), dim);
		updateState<<<(31 + dim) / 32, 32>>>(B, external, lamBeta, dim, timestep, noise, length, iterations, i);
		length += dim;
	}
}


extern "C++" void fillWeights(double *weights, int dim) {
	// allocate weight connections matrix on the GPU
	double *dev_weights;
	int size_weights = dim * dim;
	cudaMalloc(&dev_weights, size_weights * sizeof(double));

	// fill matrix and first activation vector with random values
	GPU_fill_rand(dev_weights, dim, dim);

	// change decimal values in matrix to integers between 0 and 10
	changeValues<<<(31 + size_weights) / 32, 32>>>(dev_weights, size_weights);

	// copy results to host
	cudaMemcpy(weights, dev_weights, size_weights * sizeof(double), cudaMemcpyDeviceToHost);

	//Free GPU memory
  cudaFree(dev_weights);
}

extern "C++" void fillLayers(double *layers, int dim) {
	// allocate neurons vector on the GPU
	double *dev_layers;
	int size_layers = dim;
	cudaMalloc(&dev_layers, size_layers * sizeof(double));

	// fill vector vector with random values
	GPU_fill_rand(dev_layers, dim, 1);

	// copy results to host
	cudaMemcpy(layers, dev_layers, dim * sizeof(double), cudaMemcpyDeviceToHost);

	//Free GPU memory
  cudaFree(dev_layers);
}


// external function defined in MatrixMultiplication.h
extern "C++" void matrixMultiplication(double *layers, double *weights, double *external, double *lamBeta, int dim, int iterations, float timestep, double noise) {
	int size_weights = dim * dim;
	int size_layers = dim * iterations + dim;
	int size_external = size_layers * dim - dim;
  int size_lamBeta = dim * 2;

	// allocate arrays on device
	double *dev_layers, *dev_weights, *dev_external, *dev_lamBeta;
	cudaMalloc(&dev_layers, size_layers * sizeof(double));
	cudaMalloc(&dev_external, size_external * sizeof(double));
	cudaMalloc(&dev_weights, size_weights * sizeof(double));
  cudaMalloc(&dev_lamBeta, dim * 2 * sizeof(double));

	// copy arrays to GPU
	cudaMemcpy(dev_layers, layers, size_layers * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_external, external, size_external * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weights, weights, size_weights * sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_lamBeta, lamBeta, size_lamBeta * sizeof(double),cudaMemcpyHostToDevice);

	// create handle to the cuBLAS library context
	cublasHandle_t handle;
  cublasCreate(&handle);

	gpu_blas_mmul(handle, iterations, timestep, noise, dev_weights, dev_layers, dev_external, dev_lamBeta, dim, size_layers);

	// destroy handle
	cublasDestroy(handle);

	// copy results to host
	cudaMemcpy(layers, dev_layers, size_layers * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(external, dev_external, size_layers * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(weights, dev_weights, size_weights * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lamBeta, dev_lamBeta, size_lamBeta * sizeof(double), cudaMemcpyDeviceToHost);

  //Free GPU memory
  cudaFree(dev_layers);
  cudaFree(dev_weights);
	cudaFree(dev_external);
  cudaFree(dev_lamBeta);
}
