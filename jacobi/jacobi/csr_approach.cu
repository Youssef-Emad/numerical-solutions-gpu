#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* concat(char*,char*);
__global__ void jacobiCsr(float *x, const float *diagonal_values , const float * values, const int *rowPtr ,const int *colIdx,const float *y, const int size)
{
	const int index = threadIdx.x;

	if (index < size)
	{
		int row_start = rowPtr[index];
		int row_end = rowPtr[index +1];

		float sum = 0 ;

		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = row_start ; i< row_end ; i++)
			{
				sum += values[i] * x[colIdx[i]];
			}
			x[index] =( y[index] - sum )/diagonal_values[index] ;
			sum = 0 ;
			__syncthreads();	
		}
	}
}

__global__ void jacobiCsrShared(float *x, const float *diagonal_values , const float * values, const int *rowPtr ,const int *colIdx,const float *y, const int size)
{
    const int index = threadIdx.x;

	 if (index < size)
	{
		__shared__ float shared_diagonal_values[24] ;
		__shared__ float shared_values[48];
		__shared__ int shared_colIdx[48];
		__shared__ float shared_x[24];
		__shared__ float shared_y[24];

		shared_values[2*index] = values[2*index];
		shared_values[2*index+1] = values[2*index+1];
		shared_colIdx[2*index] = colIdx[2*index];
		shared_colIdx[2*index+1] = colIdx[2*index+1];
		shared_diagonal_values[index] = diagonal_values[index];
		shared_y[index] = y[index];
		shared_x[index] = 0;

		int row_start = rowPtr[index];
		int row_end = rowPtr[index +1];

		float sum = 0 ;
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = row_start ; i< row_end ; i++)
			{
				sum += shared_values[i] * shared_x[shared_colIdx[i]];
			}
			shared_x[index] =(shared_y[index] - sum )/shared_diagonal_values[index] ;
			sum = 0 ;
			__syncthreads();	
		}
		x[index] = shared_x[index];
	}
}


void jacobiCsrOne(const int  size , char* file_name)
{
	//initialize our test cases
	
	float *values = (float *)malloc(2 * size * sizeof(float));
	float *diagonal_values = (float *)malloc(size * sizeof(float));
	int *rowPtr = (int *)malloc((size + 1) * sizeof(int));
	int *colIdx = (int *)malloc(2 * size * sizeof(int));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));

	char* diagonal_values_file_name = concat(file_name,"/csr_diagonal/csr_diagonal_value.txt") ;
	char* non_diagonal_values_file_name = concat(file_name,"/csr_diagonal/csr_non_diagonal_value.txt");
	char* rowPtr_file_name = concat(file_name,"/csr_diagonal/csr_rowPtr.txt");
	char* colIdx_file_name = concat(file_name,"/csr_diagonal/csr_colIdx.txt");
	char* y_file_name = concat(file_name,"/right_hand_side.txt");
	char* output_file_name = concat(file_name,"/output.txt");

	FILE *diagonal_values_file = fopen(diagonal_values_file_name, "r");
	FILE *non_diagonal_values_file = fopen(non_diagonal_values_file_name, "r");
	FILE *rowPtr_file = fopen(rowPtr_file_name, "r");
	FILE *colIdx_file = fopen(colIdx_file_name, "r");
	FILE *y_file = fopen(y_file_name, "r");
	FILE *output_file = fopen(output_file_name, "r");

	for (int i = 0 ; i < size ; i++)
	{
		fscanf(diagonal_values_file, "%f", &diagonal_values[i]);	
		fscanf(y_file, "%f", &y[i]);	
		fscanf(output_file, "%f", &output[i]);
		fscanf(rowPtr_file, "%d", &rowPtr[i]);
		x[i] = 0 ;
		if (i == size -1 )
		{
			fscanf(rowPtr_file, "%d", &rowPtr[i+1]);
		}
	}

	for (int i = 0 ; i< 2*size ; i++)
	{
		fscanf(colIdx_file, "%d", &colIdx[i]);
		fscanf(non_diagonal_values_file, "%f", &values[i]);	
	}
	
	float *dev_values = 0 ;
	float *dev_diagonal_values = 0 ;
	int *dev_rowPtr = 0 ;
	int *dev_colIdx = 0 ;
	float *dev_x = 0 ;
	float *dev_y = 0 ;

    cudaSetDevice(0);
	
    // Allocate GPU buffers
    cudaMalloc((void**)&dev_x, size * sizeof(float));
    cudaMalloc((void**)&dev_diagonal_values, size * sizeof(float));
    cudaMalloc((void**)&dev_values,  2 * size * sizeof(float));
	cudaMalloc((void**)&dev_rowPtr,  (size+1) * sizeof(int));
	cudaMalloc((void**)&dev_colIdx,  2 * size * sizeof(int));
	cudaMalloc((void**)&dev_y, size * sizeof(float));
   
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_diagonal_values, diagonal_values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_values, values,  2 * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_colIdx, colIdx,  2 * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_rowPtr, rowPtr, ( size+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    
	// Launch a kernel on the GPU with one thread for each row
	jacobiCsrShared<<<ceil(size/32.0), 32>>>(dev_x, dev_diagonal_values , dev_values , dev_rowPtr,dev_colIdx , dev_y , size);
	
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	 cudaDeviceSynchronize();
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	printf("%f\n",x[23]);
	cudaFree(dev_x);
	cudaFree(dev_y);
    cudaFree(dev_diagonal_values);
    cudaFree(dev_values);
    cudaFree(dev_rowPtr);
	cudaDeviceReset();
}
