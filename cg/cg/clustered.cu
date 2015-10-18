#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

char* concat(char *s1, char *s2);

__global__ void cg_variable_start(float* a , float* x,float * b ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	int local_index = threadIdx.x ;
	int block_index = blockIdx.x ;

	__shared__ float shared_r_squared[1024] ;
	__shared__ float shared_p_sum[1024] ;
	__shared__ float r[1024] ;
	float local_r ;

	shared_r_squared[local_index] = 0 ;
	shared_p_sum[local_index] = 0;
	__syncthreads() ;
	
	if (index < size)
	{
		float sum = 0 ;
		
		for (int i = 0 ; i<3 ; i++)
		{
			sum += a[3 * index  + i] * x[3 * block_index + i] ;
		}
		
		local_r = b[index] - sum ;	
		r[local_index] = local_r ;
		__syncthreads() ;

		for (int i = 0 ; i<3 ; i++)
		{
			shared_p_sum[local_index] += a[3*index  + i] * r[3* block_index + i] ;
		}
		__syncthreads() ;

		shared_r_squared[local_index] = local_r * local_r ;
		shared_p_sum[local_index] = shared_p_sum[local_index] * local_r ;
	}
	
	__syncthreads() ;

	for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
	{	
		if (threadIdx.x < s)
		{
			shared_r_squared[local_index] = shared_r_squared[local_index] + shared_r_squared[local_index +s] ;
			shared_p_sum[local_index] = shared_p_sum[local_index] + shared_p_sum[local_index +s] ;
			__syncthreads() ;
		}
			
	}	
	__syncthreads();

	if (index < size)
	{
		float alpha = shared_r_squared[0]/shared_p_sum[0] ;
		x[index] = x[index] + alpha * local_r ;	
	}

}

__global__ void cg_zero_start(float* a , float* x,float * b ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	int local_index = threadIdx.x ;
	int block_index = blockIdx.x ;
	
	__shared__ float shared_r_squared[1024] ;
	__shared__ float shared_p_sum[1024] ;
	float local_b ;
	
	shared_r_squared[local_index] = 0 ;
	shared_p_sum[local_index] = 0;
	__syncthreads() ;
	
	if (index < size)
	{
		local_b = b[index] ;

		for (int i = 0 ; i<3 ; i++)
		{
			shared_p_sum[local_index] += a[3*index  + i] * b[3* block_index + i] ;
		}
		__syncthreads() ;

		shared_r_squared[local_index] = local_b * local_b ;
		shared_p_sum[local_index] = shared_p_sum[local_index] * local_b ;
	}
	
	__syncthreads() ;

	for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
	{	
		if (threadIdx.x < s)
		{
			shared_r_squared[local_index] = shared_r_squared[local_index] + shared_r_squared[local_index +s] ;
			shared_p_sum[local_index] = shared_p_sum[local_index] + shared_p_sum[local_index +s] ;
			__syncthreads() ;
		}
			
	}	
	__syncthreads();

	if (index < size)
	{
		float alpha = shared_r_squared[0]/shared_p_sum[0] ;
		x[index] = x[index] + alpha * local_b ;	
	}

}
