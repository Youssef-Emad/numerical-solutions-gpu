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

void cg_clustered(const int size , char* file_name)
{
	//initialize our test cases

	float *values = (float *)malloc(3 * size * sizeof(float));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));
	

	char* values_file_name = concat(file_name,"/basic/values.txt") ;
	char* y_file_name = concat(file_name,"/right_hand_side.txt");
	char* output_file_name = concat(file_name,"/output.txt");

	FILE *values_file = fopen(values_file_name, "r");
	FILE *y_file = fopen(y_file_name, "r");
	FILE *output_file = fopen(output_file_name, "r");

	for (int i = 0 ; i < size ; i++)
	{	
		fscanf(y_file, "%f", &y[i]);
		fscanf(output_file, "%f", &output[i]);
		x[i] = 0 ;
	}

	for (int i = 0 ; i< 3 * size ; i++)
	{
		fscanf(values_file, "%f", &values[i]);
	}
	
	float* dev_values = 0;
	float* dev_y = 0;
	float* dev_x = 0;
	
    cudaSetDevice(0);
	
    // Allocate GPU buffers
    cudaMalloc((void**)&dev_values, 3 * size * sizeof(float));
    cudaMalloc((void**)&dev_y, size * sizeof(float));
    cudaMalloc((void**)&dev_x, size * sizeof(float));
	
   
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_values, values, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each row.
	cg_zero_start<<<350,698>>>(dev_values , dev_x,dev_y ,size) ;
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);

	printf("%f\n",x[0]);
	printf("%f\n",x[1]);
	printf("%f\n",x[2]);
	printf("%f\n",x[size -2]);
	printf("%f\n",x[size -1]);
	
	cudaFree(dev_values);
	cudaFree(dev_y);
	cudaFree(dev_x);
	
	cudaDeviceReset();
	system("pause");
}

int main()
{
	cg_clustered(244300,"C:/Users/youssef/Desktop/numerical-solutions-gpu/cg/cg/test_cases/244300");
	return 1 ;
}