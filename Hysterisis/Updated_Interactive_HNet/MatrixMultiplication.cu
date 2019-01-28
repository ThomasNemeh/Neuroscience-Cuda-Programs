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

// basic file operations
#include <iostream>
#include <fstream>
using namespace std;

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
__global__ void squeeze(float *B, int dim, int length, float L, float M) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
	if (index < length + dim) {
		B[index] = 1 / (1 + expf(-1 * L * (B[index] - M)));
	}
}

// perform the matrix multiplication operation
// Param: handle = handle to the cuBLAS library context. iterations = number of times we multiply activation vector by matrix
//        A = matrix. B = array of activation vectors calculated so far. dim = length & width of square matrix. L, M = parameter for squeeze function
void gpu_blas_mmul(cublasHandle_t &handle, const float *A, float *B, const int dim, float L, float M) {
    const float alf = 1; // scalar used for multiplication
    const float bet = 0; // scalar used for multiplication
    const float *alpha = &alf;
    const float *beta = &bet;
	int length = 0;
	
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, 1, dim, alpha, A, dim, B, dim, beta, (B + length), dim);
	squeeze<<<(31 + dim) / 32, 32>>>(B, dim, length, L, M);
	length += dim;
}

// external function defined in RandomMatrixMultiplication.h
extern "C++" void matrixMultiplication(float *layers, float *weights, int dim, float L, float M) {
	// allocate arrays on device
	float *dev_layers, *dev_weights;
	cudaMalloc(&dev_layers, dim * sizeof(float));
	cudaMalloc(&dev_weights, dim * dim * sizeof(float));
	
	// copy arrays to GPU
	cudaMemcpy(dev_layers, layers, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weights, weights, dim * dim * sizeof(float),cudaMemcpyHostToDevice);
	
	// create handle to the cuBLAS library context
	cublasHandle_t handle;
    cublasCreate(&handle);
	
	gpu_blas_mmul(handle, 1, dev_layers, dev_weights, dim, L, M);
	
	// destroy handle
	cublasDestroy(handle);
	
	// copy results to host
	cudaMemcpy(layers, dev_layers, dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(weights, dev_weights, dim * dim * sizeof(float),cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(dev_layers);
    cudaFree(dev_weights);
    
}
