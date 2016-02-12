
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include<cublas_v2.h>

cublasHandle_t handle;

char* concat(char *s1, char *s2);

__global__ void ap_multiplication(float * values ,int * indeces,float* r ,float * p_sum ,int size) 
{
	int index = blockDim.x * blockIdx.x + threadIdx.x ;

	p_sum[index] = 0;
	__syncthreads() ;
	if (index < size)
	{
		for (int i = 0 ; i<3 ; i++)
		{
			p_sum[index] += values[3*index  + i] * r[indeces[3*index + i]] ;
		}
		__syncthreads() ;
	}
}

__global__ void alpha_calculation(float * r_squared ,float * p_sum,float* alpha) 
{
	alpha[0] = r_squared[0]/p_sum[0] ;
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

void cg_cublas_zero_start(int size , int number_of_blocks , int number_of_threads,float * values , int * indeces ,float * x , float* y , float * r_squared ,float* p_sum,float* ap_total)
{
	cublasSdot(handle, size, y, 1, y, 1, r_squared);
	ap_multiplication<<<number_of_blocks,number_of_threads>>>(values,indeces,y,p_sum,size);
	cublasSdot(handle, size,y , 1,p_sum , 1, ap_total);
	alpha_calculation<<<1,1>>>(r_squared,ap_total,r_squared);
	x_calculation<<<number_of_blocks,number_of_threads>>>(x ,y,r_squared , size);
}

void cg_cublas(const int size , char* file_name)
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
	float* dev_ap_total = 0;
	int number_of_blocks = 1;
	int number_of_threads = 12;

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

	cublasCreate(&handle);
	cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

    cudaSetDevice(0);
	
    // Allocate GPU buffers
    cudaMalloc((void**)&dev_values, 3 * size * sizeof(float));
	cudaMalloc((void**)&dev_indeces, 3 * size * sizeof(int));
    cudaMalloc((void**)&dev_y, size * sizeof(float));
    cudaMalloc((void**)&dev_x, size * sizeof(float));
	cudaMalloc((void**)&dev_r, size * sizeof(float));
	cudaMalloc((void**)&dev_r_squared, size * sizeof(float));
	cudaMalloc((void**)&dev_p_sum, size * sizeof(float));
	cudaMalloc((void**)&dev_ap_total, size * sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_values, values, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_indeces, indeces, 3 * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
    // Launch a kernel on the GPU with one thread for each row.
	cg_cublas_zero_start(size , number_of_blocks , number_of_threads,dev_values , dev_indeces ,dev_x , dev_y , dev_r_squared ,dev_p_sum,dev_ap_total);
	cudaEventRecord(stop);
   
    cudaMemcpyAsync(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("kernel time = %f\n",milliseconds);
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.

	cudaFree(dev_values);
	cudaFree(dev_indeces) ;
	cudaFree(dev_y);
	cudaFree(dev_x);
	cudaFree(dev_r) ;
	cudaFree(dev_r_squared) ;
	cudaFree(dev_p_sum) ;
	cudaDeviceReset();
}

char* concat(char *s1, char *s2)
{
    char *result = (char *)malloc(strlen(s1)+strlen(s2)+1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}
