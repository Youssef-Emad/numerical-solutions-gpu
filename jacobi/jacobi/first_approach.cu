#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void jacobiFirst(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces,const float *y, const int size);

__global__ void jacobiOne(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
    const int index = threadIdx.x;
	float sum = 0 ;

	if (index < size)
	{
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = 0 ; i<2 ; i++)
			{
				sum += non_diagonal_values[2*index + i]  * x[indeces[2*index + i]] ;
			}
			x[index] = (y[index] - sum )/diagonal_values[index];
			sum = 0 ;
			__syncthreads();	
		}
	}
}

__global__ void jacobiOneShared(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
    const int index = threadIdx.x;
	__shared__ float shared_diagonal_values[24] ;
	__shared__ float shared_non_diagonal_values[48];
	__shared__ int shared_indeces[48];
	__shared__ float shared_y[24];
	__shared__ float shared_x[24];

	shared_diagonal_values[index] = diagonal_values[index];
	shared_non_diagonal_values[2*index] = non_diagonal_values[2*index];
	shared_non_diagonal_values[2*index+1] = non_diagonal_values[2*index+1];
	shared_indeces[2*index] = indeces[2*index];
	shared_indeces[2*index+1] = indeces[2*index+1];
	shared_y[index] = y[index];
	shared_x[index] = x[index];

	float sum = 0 ;
	if (index < size)
	{
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = 0 ; i<2 ; i++)
			{
				sum += shared_non_diagonal_values[2*index + i]  * shared_x[shared_indeces[2*index + i]] ;
			}
			shared_x[index] = (shared_y[index] - sum )/shared_diagonal_values[index];
			sum = 0 ;
			__syncthreads();	
		}
		x[index] = shared_x[index];
	}
}

__global__ void jacobiOneSharedAndLocal(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
    const int index = threadIdx.x;
	float local_diagonal_value ;
	float local_non_diagonal_values[2];
	int local_indeces[2];
	float local_y;
	__shared__ float shared_x[24];

	local_diagonal_value = diagonal_values[index];
	local_non_diagonal_values[0] = non_diagonal_values[2*index];
	local_non_diagonal_values[1] = non_diagonal_values[2*index+1];
	local_indeces[0] = indeces[2*index];
	local_indeces[1] = indeces[2*index+1];
	local_y = y[index];
	shared_x[index] = x[index];

	float sum = 0 ;
	if (index < size)
	{
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = 0 ; i<2 ; i++)
			{
				sum += local_non_diagonal_values[i]  * shared_x[local_indeces[i]] ;
			}
			
			shared_x[index] = (local_y - sum )/local_diagonal_value;
			sum = 0 ;
			__syncthreads();	
		}
		x[index] = shared_x[index];
	}
}

void jacobiFirst(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces,const float *y, const int size)
{
    float *dev_non_diagonal_values = 0;
	float *dev_diagonal_values = 0;
    int *dev_indeces = 0;
	float *dev_y = 0 ;
    float *dev_x = 0;

    cudaSetDevice(0);
	

    // Allocate GPU buffers
    cudaMalloc((void**)&dev_x, size * sizeof(float));
    cudaMalloc((void**)&dev_non_diagonal_values, 2 * size * sizeof(float));
    cudaMalloc((void**)&dev_indeces, 2 * size * sizeof(int));
	cudaMalloc((void**)&dev_y, size * sizeof(float));
   
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_diagonal_values, diagonal_values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_non_diagonal_values, non_diagonal_values, 2 * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_indeces, indeces, 2 * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch a kernel on the GPU with one thread for each element.
    jacobiOneSharedAndLocal<<<1, size>>>(dev_x, dev_diagonal_values , dev_non_diagonal_values , dev_indeces , dev_y , size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
   cudaDeviceSynchronize();
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
    
Error:
    cudaFree(dev_x);
	cudaFree(dev_y);
    cudaFree(dev_diagonal_values);
    cudaFree(dev_non_diagonal_values);
    cudaFree(dev_indeces);
}