#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

char* concat(char *s1, char *s2);

__global__ void r_calculation(float* a , int * indeces , float* b , float* x,float * r  ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	
	if (index < size)
	{
		float sum = 0 ;
		
		for (int i = 0 ; i<3 ; i++)
		{
			sum += a[3*index  + i] * x[indeces[3*index + i]] ;
		}
		
		r[index] = b[index] - sum ;	
	}
	
}

__global__ void r_initial_sum(float* a , int * indeces , float* x,float * r ,float * r_squared ,float * p_sum ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	int local_index = threadIdx.x ;
	
	__shared__ float shared_r_squared[1024] ;
	__shared__ float shared_p_sum[1024] ;

	shared_r_squared[local_index] = 0 ;
	shared_p_sum[local_index] = 0;
	__syncthreads() ;
	
	if (index < size)
	{
		for (int i = 0 ; i<3 ; i++)
		{
			shared_p_sum[local_index] += a[3*index  + i] * r[indeces[3*index + i]] ;
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

	if (threadIdx.x == 0)
	{
		r_squared[blockIdx.x] = shared_r_squared[0];
		p_sum[blockIdx.x] = shared_p_sum[0];
		__syncthreads() ;
	}
}

__global__ void r_final_sum_and_alpha_calculation(float * r_squared ,float * p_sum ,int size) 
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
		if (index < s)
		{
			shared_r_squared[index] = shared_r_squared[index] + shared_r_squared[index +s] ;
			shared_p_sum[index] = shared_p_sum[index] + shared_p_sum[index +s] ;
			__syncthreads() ;
		}	
	}	
	if(threadIdx.x == 0)
	{
		//alpha
		r_squared[blockIdx.x] = shared_r_squared[0]/shared_p_sum[0] ;
		
	}
}

__global__ void x_calculation(float * x ,float * r,float * r_squared ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;
	if (index < size)
	{
		float alpha = r_squared[0] ;
		x[index] = x[index] + alpha * r[index] ;
	}
}

void cg_zero_start(int size , int number_of_blocks , int number_of_threads,float * values , int * indeces ,float * x , float* y , float * r_squared ,float* p_sum)
{
	r_initial_sum<<<number_of_blocks,number_of_threads>>>(values , indeces , x, y , r_squared , p_sum , size) ;
	r_final_sum_and_alpha_calculation<<<1,number_of_blocks>>>(r_squared ,p_sum ,number_of_blocks);
	x_calculation<<<number_of_blocks,number_of_threads>>>(x ,y,r_squared , size);
}

void cg_variable_start(int size , int number_of_blocks , int number_of_threads,float * values , int * indeces ,float * x , float* y , float * r , float * r_squared ,float* p_sum)
{
	r_calculation<<<number_of_blocks,number_of_threads>>>(values , indeces , y ,  x, r , size) ;
	r_initial_sum<<<number_of_blocks,number_of_threads>>>(values , indeces , x, r , r_squared , p_sum , size) ;
	r_final_sum_and_alpha_calculation<<<1,number_of_blocks>>>(r_squared ,p_sum ,number_of_blocks);
	x_calculation<<<number_of_blocks,number_of_threads>>>( x ,r,r_squared , size);
}


void cg(const int size , char* file_name)
{
	//initialize our test cases

	float *values = (float *)malloc(3 * size * sizeof(float));
	int *indeces = (int *)malloc(3 * size * sizeof(int));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));
	float *r_sqaured = (float *)malloc(size * sizeof(float));
	float *p_sum = (float *)malloc(size * sizeof(float));

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

	for (int i = 0 ; i< size; i++)
	{
		r_sqaured[i] = 0 ;
		p_sum[i] = 0 ;
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
	int number_of_blocks ;
	int number_of_threads ;

	if (size < 671088)
	{
		number_of_blocks = sqrt(size * 1.0) * 0.8 ;
		number_of_threads = ceil((size * 1.0)/number_of_blocks) ;
	} 
	else 
	{
		number_of_blocks = ceil(size/1024.0) ;
		number_of_threads = 1024 ;	
	}
	
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
	cg_zero_start(size , number_of_blocks , number_of_threads,dev_values , dev_indeces ,dev_x , dev_y , dev_r_squared ,dev_p_sum);
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
	cudaFree(dev_indeces) ;
	cudaFree(dev_y);
	cudaFree(dev_x);
	cudaFree(dev_r) ;
	cudaFree(dev_r_squared) ;
	cudaFree(dev_p_sum) ;
	cudaDeviceReset();
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
	cg(244300,"C:/Users/youssef/Desktop/numerical-solutions-gpu/cg/cg/test_cases/244300");
	return 1 ;
}