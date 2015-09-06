#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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

