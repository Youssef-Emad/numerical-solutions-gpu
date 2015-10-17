#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

char* concat(char *s1, char *s2);


__global__ void cg_full_global(float* a , int * indeces , float* b , float* x,float * r ,float * r_squared ,float * p_sum ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	/*float local_a[3] = {a[3*index],a[3*index + 1],a[3*index + 2]} ;
	int local_indeces[3]  = {indeces[3*index],indeces[3*index + 1],indeces[3*index + 2]} ;*/

	if (index < size)
	{
		float sum = 0 ;
		
		for (int i = 0 ; i<3 ; i++)
		{
			sum += a[3*index  + i] * x[indeces[3*index + i]] ;
		}
		
		float local_r = b[index] - sum ;	
		r[index] = local_r;

		p_sum[index] = 0 ;
		__syncthreads();

		for (int i = 0 ; i<3 ; i++)
		{
			p_sum[index] += a[3*index  + i] * r[indeces[3*index + i]] ;
		}
		
		//calc alpha
		r_squared[index] = local_r * local_r ;
		p_sum[index] = p_sum[index] * local_r ;
		
		//sum inside block
		for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
		{	
			if (threadIdx.x < s/2)
			{
				// summation of r*rT
				r_squared[index] = r_squared[index] + r_squared[index + s] ;
				//summation of r*a*rT
				p_sum[index] = p_sum[index] +  p_sum[index + s] ;
			}
			__syncthreads();
		}
		
		// sum between blocks - 1024 or 2048 blocks only
		int max_index_of_needed_blocks = gridDim.x/1025 ;

		if (blockIdx.x <= max_index_of_needed_blocks && gridDim.x > 1)
		{
			if(threadIdx.x == 0)
			{
				r_squared[blockIdx.x] = r_squared[blockIdx.x * blockDim.x] ;
				p_sum[blockIdx.x] = p_sum[blockIdx.x * blockDim.x] ;
			}
				
			for (unsigned int s = gridDim.x/2 ; s> 0 ; s >>= 1)
			{	
				if (index < s/2)
				{
					// summation of r*rT
					r_squared[index] = r_squared[index] + r_squared[index +  s] ;
					//summation of r*a*rT
					p_sum[index] = p_sum[index] +  p_sum[index +  s] ;
				}
				__syncthreads();
			}
		}
		
		float alpha = r_squared[0]/p_sum[0] ;
		x[index] = x[index] + alpha * local_r ;
	}
}

__global__ void cg_one(float* a , int * indeces , float* b , float* x,float * r ,float * r_squared ,float * p_sum ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	int local_index = threadIdx.x ;
	/*float local_a[3] = {a[3*index],a[3*index + 1],a[3*index + 2]} ;
	int local_indeces[3]  = {indeces[3*index],indeces[3*index + 1],indeces[3*index + 2]} ;*/
	__shared__ float shared_r_squared[1024] ;
	__shared__ float shared_p_sum[1024] ;

	shared_r_squared[index] = 0 ; // for extra numbers
	shared_p_sum[index] = 0 ;
	
	if (index < size)
	{
		float sum = 0 ;
		
		for (int i = 0 ; i<3 ; i++)
		{
			sum += a[3*index  + i] * x[indeces[3*index + i]] ;
		}
		
		float local_r = b[index] - sum ;	
		r[index] = local_r;


		for (int i = 0 ; i<3 ; i++)
		{
			shared_p_sum[index] += a[3*index  + i] * r[indeces[3*index + i]] ;
		}
		
		shared_r_squared[local_index] = local_r * local_r ;
		shared_p_sum[local_index] = shared_p_sum[index] * local_r ;
	}
	
	__syncthreads() ;

	for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
	{	
		if (threadIdx.x < s)
		{
			shared_r_squared[threadIdx.x] = shared_r_squared[threadIdx.x] + shared_r_squared[threadIdx.x +s] ;
			shared_p_sum[threadIdx.x] = shared_p_sum[threadIdx.x] + shared_p_sum[threadIdx.x +s] ;
			__syncthreads() ;
		}
			
	}	
	r_squared[blockIdx.x] = shared_r_squared[0] ;
	p_sum[blockIdx.x] = shared_p_sum[0] ;

}

__global__ void cg_two(float * r_squared ,float * p_sum ,int size) 
{
	int index = threadIdx.x ;
	__shared__ float shared_r_squared[1024] ;
	__shared__ float shared_p_sum[1024] ;

	if (index < size)
	{
		shared_r_squared[index] = r_squared[index]  ;
		shared_p_sum[index] = p_sum[index]  ;
	} else
	{
		shared_r_squared[index] = 0 ;
		shared_p_sum[index] = 0 ;
	}
	__syncthreads() ;


	for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
	{	
		if (threadIdx.x < s)
		{
			shared_r_squared[threadIdx.x] = shared_r_squared[threadIdx.x] + shared_r_squared[threadIdx.x +s] ;
			shared_p_sum[threadIdx.x] = shared_p_sum[threadIdx.x] + shared_p_sum[threadIdx.x +s] ;
			__syncthreads() ;
		}	
	}	
	if(threadIdx.x == 0)
	{
		//alpha
		r_squared[blockIdx.x] = shared_r_squared[0]/shared_p_sum[0] ;
	}
}

__global__ void cg_three(float * x ,float * r,float * r_squared ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	float alpha = r_squared[0] ;
	x[index] = x[index] + alpha * r[index] ;
}

void cg(const int size , char* file_name)
{
	//initialize our test cases

	float *values = (float *)malloc(3 * size * sizeof(float));
	int *indeces = (int *)malloc(3 * size * sizeof(int));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));

	char* values_file_name = concat(file_name,"/basic/values.txt") ;
	char* indeces_file_name = concat(file_name,"/basic/indeces.txt");
	char* y_file_name = concat(file_name,"/right_hand_side.txt");
	char* output_file_name = concat(file_name,"/output.txt");

	FILE *values_file = fopen(values_file_name, "r");
	FILE *indeces_file = fopen(indeces_file_name, "r");
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
		fscanf(indeces_file, "%d", &indeces[i]);	
	}
	
	float* dev_values = 0;
	int* dev_indeces = 0 ;
	float* dev_y = 0;
	float* dev_x = 0;
	float* dev_r = 0 ;
	float* dev_r_squared = 0 ;
	float* dev_p_sum = 0;

	int fraction = ceil(size/1024.0) ;
	int number_of_blocks = 100 ;

    cudaSetDevice(0);
	
    // Allocate GPU buffers
    cudaMalloc((void**)&dev_values, 3 * size * sizeof(float));
	cudaMalloc((void**)&dev_indeces, 3 * size * sizeof(int));
    cudaMalloc((void**)&dev_y, size * sizeof(float));
    cudaMalloc((void**)&dev_x, size * sizeof(float));
	cudaMalloc((void**)&dev_r, size * sizeof(float));
	cudaMalloc((void**)&dev_r_squared, number_of_blocks * sizeof(float));
	cudaMalloc((void**)&dev_p_sum, number_of_blocks * sizeof(float));
   
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_values, values, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_indeces, indeces, 3 * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each row.
	cg_one<<<2,12>>>(dev_values , dev_indeces , dev_y ,  dev_x, dev_r , dev_r_squared , dev_p_sum , size) ;
	cg_two<<<1,2>>>(dev_r_squared ,dev_p_sum ,size);
	cg_three<<<2,12>>>( dev_x ,dev_r,dev_r_squared , size);
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
	cudaDeviceReset();
	cudaFree(dev_values);
	cudaFree(dev_indeces) ;
	cudaFree(dev_y);
	cudaFree(dev_x);
	cudaFree(dev_r) ;
	cudaFree(dev_r_squared) ;
	cudaFree(dev_p_sum) ;

	system("pause");
}

char* concat(char *s1, char *s2)
{
    char *result = (char *)malloc(strlen(s1)+strlen(s2)+1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main()
{
	cg(24,"C:/Users/youssef/Desktop/numerical-solutions-gpu/cg/cg/test_cases/24");
	return 1 ;
}