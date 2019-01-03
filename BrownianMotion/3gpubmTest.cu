#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "book.h"	
#include <vector>


int *crossTimes = nullptr;

//Function to generate brownian path, which is stored in results. Executes on the GPU, hence the __global__ identifier
__global__ void randomWalk(double *results, int *crossTimes, int T, int N, double drift, int numSims, double lowerThreshold, double upperThreshold, int deviceID) {

	int crossTimeIndex = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (crossTimeIndex < numSims) {
	
		curandState_t state;
		curand_init (blockIdx.x * (1000 * deviceID) + threadIdx.x + clock64(), 0, 0, &state);
		double random;
		int start = (threadIdx.x + blockIdx.x * blockDim.x) * N;  
	
		crossTimes[crossTimeIndex] = 0;
		results[start] = 0.0;
		bool crossed = false;
	
		for (int j = start + 1; j < start + N; j++) {
			random = curand_normal_double(&state);
			// results[j] = results[j-1] + random * sqrt((double) T / N);
			results[j] = results[j-1] + drift * ((double) T / N) + random * sqrt((double) T / N);
			if (!crossed && results[j] >= upperThreshold) {
				crossTimes[crossTimeIndex] = j - start;
				crossed = true;
			}
			else if (!crossed && results[j] <= lowerThreshold) {
				crossTimes[crossTimeIndex] = -1 * (j - start);
				crossed = true;
			}
		}
	
	}
	
	
	/*
	Generate 2 doubles at once. Test later to see if this is more efficient:
	double curand_normal2_double (state);  
	*/
	
}

struct DataStruct {
	int deviceID;
	int sims;
	double *resultArray;
	int *crossArray;
	int N;
	double T;
	double drift;
	double lowerThreshold;
	double upperThreshold;
};

void* routine(void *voidData) {
	DataStruct *data = (DataStruct*)voidData;
	HANDLE_ERROR(cudaSetDevice(data->deviceID));
	
	int sims = data->sims;
	
	double *dev_results;
	double *partialResults = (double*)malloc(sims * data->N * sizeof(double));
	
	int *dev_crossTimes;
	int *partialCrossTimes = (int*)malloc(sims * sizeof(int));
	
	
	HANDLE_ERROR(cudaMalloc(&dev_results, data->N * sims * sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&dev_crossTimes, sims * sizeof(int)));
	
	//Calculate number of blocks and threads for global function
	int numBlocks = (511 + sims) / sims;
	int numThreads = 512;
	
	//Call GPU function
	randomWalk<<<numBlocks, numThreads>>>(dev_results, dev_crossTimes, data->T, data->N, data->drift, sims, data->lowerThreshold, data->upperThreshold, data->deviceID);
	
	HANDLE_ERROR(cudaMemcpy(partialResults, dev_results , data->N * sims * sizeof(double), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(partialCrossTimes, dev_crossTimes , sims * sizeof(int), cudaMemcpyDeviceToHost));
	
	data->resultArray = partialResults;
	data->crossArray = partialCrossTimes;
	
	HANDLE_ERROR(cudaFree(dev_results));
	HANDLE_ERROR(cudaFree(dev_crossTimes));
	
	return 0;
}

int main() {
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start,0));

	int N = 20;
	double T = 1;
	int numSims = 1;
	double drift = 1;
	double lowerThreshold = -1;
	double upperThreshold = 1;

	DataStruct data[3];
	data[0].deviceID = 0;
	data[0].sims = numSims / 3 + numSims % 3;
	data[0].N = N;
	data[0].T = T;
	data[0].drift = drift;
	data[0].lowerThreshold = lowerThreshold;
	data[0].upperThreshold = upperThreshold;
	if (numSims > 2) {
		data[1].deviceID = 0;
		data[1].sims = numSims / 3;
		data[1].N = N;
		data[1].T = T;
		data[1].drift = drift;
		data[1].lowerThreshold = lowerThreshold;
		data[1].upperThreshold = upperThreshold;
		data[2].deviceID = 0;
		data[2].sims = numSims / 3;
		data[2].N = N;
		data[2].T = T;
		data[2].drift = drift;
		data[2].lowerThreshold = lowerThreshold;
		data[2].upperThreshold = upperThreshold;
	
		CUTThread thread = start_thread(routine, &(data[0]));
		CUTThread thread2 = start_thread(routine, &(data[1]));
		routine(&(data[2]));
		end_thread(thread);
		end_thread(thread2);
	} else {
		routine(&(data[0]));
	}
	
	
	double *results = new double[N * numSims];
	crossTimes = new int[numSims];
	
	if (numSims > 2) {
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
	}
	else {
		results = data[0].resultArray;
		crossTimes = data[0].crossArray;
	}
	
	
	
	HANDLE_ERROR(cudaEventRecord(stop,0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	//printf("Time to generate: %3.1f ms/n", elapsedTime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	
	
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


