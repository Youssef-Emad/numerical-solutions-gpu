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

