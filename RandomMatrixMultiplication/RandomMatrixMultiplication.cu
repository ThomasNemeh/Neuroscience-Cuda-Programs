//Program to multiply square matrix with activation vector, both filled with random numbers, and then to multiply the resulting vector by the matrix again- repeat for the 
//specified number of iterations.

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdlib>
#include <iostream>
using std::cout;
using std::endl;
using std::copy;

float *h_A = nullptr;
float *h_B = nullptr;

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
void print_matrix(const float *matrix, int rows, int cols) {
 
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            std::cout << matrix[j * rows + i] << " ";
        }
        std::cout << std::endl;
	}
    std::cout << std::endl;
}

// perform the sqeeze function on each element of the vector resulting from the later iteration of matrix multiplication
// Param: B = pointer to activation vector, dim = starting point of the vector results of the last iteration of matrix multiplication, 
// L and M are parameters of the squeeze function
__global__ void squeeze(float *B, int dim, int length, float L, float M) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
	if (index < length + dim) {
		B[index] = 1 / (1 + expf(-1 * L * (B[index] - M)));
	}
}

// perform the matrix multiplication operation
// Param: handle = handle to the cuBLAS library context. iterations = number of times we multiply activation vector by matrix
//        A = matrix. B = array of activation vectors calculated so far. dim = length & width of square matrix. L, M = parameter for squeeze function
void gpu_blas_mmul(cublasHandle_t &handle, int iterations, const float *A, float *B, const int dim, float L, float M) {
    const float alf = 1; // scalar used for multiplication
    const float bet = 0; // scalar used for multiplication
    const float *alpha = &alf;
    const float *beta = &bet;
	int length = 0;
	
	for (int i = 0; i < iterations; i++) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, 1, dim, alpha, A, dim, B, dim, beta, (B + length), dim);
		squeeze<<<(31 + dim) / 32, 32>>>(B, dim, length, L, M);
		length += dim;
	} 
}

// external function defined in RandomMatrixMultiplication.h
extern "C++" float *matrixMultiplication(int dim, int iterations, float L, float M) {
	int size_A = dim * dim;
	
	// allocate square matrix on host
	h_A = new float[size_A];
	//allocate array to hold activation vectors on host
    h_B = new float[dim * iterations];
	
	// allocate arrays on device
	float *dev_A, *dev_B;
	cudaMalloc(&dev_A, size_A * sizeof(float));
	cudaMalloc(&dev_B, dim * iterations * sizeof(float));
	
	// fill matrix and first activation vector with random values
	GPU_fill_rand(dev_A, dim, dim);
    GPU_fill_rand(dev_B, dim, 1);
	
	// change decimal values in matrix to integers between 0 and 10
	changeValues<<<(31 + size_A) / 32, 32>>>(dev_A, size_A);
	
	// create handle to the cuBLAS library context
	cublasHandle_t handle;
    cublasCreate(&handle);
	
	gpu_blas_mmul(handle, iterations, dev_A, dev_B, dim, L, M);
	
	// destroy handle
	cublasDestroy(handle);
	
	// copy results to host
	cudaMemcpy(h_A, dev_A, size_A * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, dev_B, dim * iterations * sizeof(float),cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(dev_A);
    cudaFree(dev_B);
	
	return h_B;
    
}

// return matrix
extern "C++" float *getMatrix() {
	return h_A;
}