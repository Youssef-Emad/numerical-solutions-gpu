#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

__global__ void cg_variable_start(float* a , float* x,float * b ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	int local_index = threadIdx.x ;
	int block_index = blockIdx.x ;
	
	__shared__ float shared_r_squared[1024] ;
	__shared__ float shared_p_sum[1024] ;
	__shared__ float r[1024] ;

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
		
		r[index] = b[index] - sum ;	
		__syncthreads() ;

		for (int i = 0 ; i<3 ; i++)
		{
			shared_p_sum[local_index] += a[3*index  + i] * r[3* block_index + i] ;
		}
		__syncthreads() ;

		shared_r_squared[local_index] = r[index] * r[index] ;
		shared_p_sum[local_index] = shared_p_sum[local_index] * r[index] ;
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
		x[index] = x[index] + alpha * r[index] ;	
	}

}

__global__ void cg_zero_start(float* a , float* x,float * b ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	int local_index = threadIdx.x ;
	int block_index = blockIdx.x ;
	
	__shared__ float shared_r_squared[1024] ;
	__shared__ float shared_p_sum[1024] ;
	
	shared_r_squared[local_index] = 0 ;
	shared_p_sum[local_index] = 0;
	__syncthreads() ;
	
	if (index < size)
	{
		for (int i = 0 ; i<3 ; i++)
		{
			shared_p_sum[local_index] += a[3*index  + i] * b[3* block_index + i] ;
		}
		__syncthreads() ;

		shared_r_squared[local_index] = b[index] * b[index] ;
		shared_p_sum[local_index] = shared_p_sum[local_index] * b[index] ;
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
		x[index] = x[index] + alpha * b[index] ;	
	}

}