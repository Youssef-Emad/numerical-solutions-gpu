#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>

char* concat(char*,char*);

/*__global__ void jacobiOne(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
    const int index = threadIdx.x;

	if (index < size)
	{
		float sum = 0 ;
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
	
	if (index < size)
	{
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
}*/

__global__ void jacobiFirstLocal(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index < size)
	{
		float local_diagonal_value ;
		float local_non_diagonal_values[2];
		int local_indeces[2];
		float local_y;

		local_diagonal_value = diagonal_values[index];
		local_non_diagonal_values[0] = non_diagonal_values[2*index];
		local_non_diagonal_values[1] = non_diagonal_values[2*index+1];
		local_indeces[0] = indeces[2*index];
		local_indeces[1] = indeces[2*index+1];
		local_y = y[index];
		
		float sum = 0 ;

		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = 0 ; i<2 ; i++)
			{
				sum += local_non_diagonal_values[i]  * x[local_indeces[i]] ;
			}
			
			x[index] = (local_y - sum )/local_diagonal_value;
			sum = 0 ;
			__syncthreads();	
		}
	}
}

void jacobiFirst(const int size , char* file_name)
{
	//initialize our test cases

	float *non_diagonal_values = (float *)malloc(2 * size * sizeof(float));
	float *diagonal_values = (float *)malloc(size * sizeof(float));
	int *indeces = (int *)malloc(2 * size * sizeof(int));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));

	char* diagonal_values_file_name = concat(file_name,"/first_approach/first_approach_diagonal_values.txt") ;
	char* non_diagonal_values_file_name = concat(file_name,"/first_approach/first_approach_non_diagonal_values.txt");
	char* indeces_file_name = concat(file_name,"/first_approach/first_approach_indeces.txt");
	char* y_file_name = concat(file_name,"/right_hand_side.txt");
	char* output_file_name = concat(file_name,"/output.txt");

	FILE *diagonal_values_file = fopen(diagonal_values_file_name, "r");
	FILE *non_diagonal_values_file = fopen(non_diagonal_values_file_name, "r");
	FILE *indeces_file = fopen(indeces_file_name, "r");
	FILE *y_file = fopen(y_file_name, "r");
	FILE *output_file = fopen(output_file_name, "r");

	for (int i = 0 ; i < size ; i++)
	{
		fscanf(diagonal_values_file, "%f", &diagonal_values[i]);	
		fscanf(y_file, "%f", &y[i]);	
		fscanf(output_file, "%f", &output[i]);
		x[i] = 0 ;
	}

	for (int i = 0 ; i< 2*size ; i++)
	{
		fscanf(indeces_file, "%d", &indeces[i]);	
		fscanf(non_diagonal_values_file, "%f", &non_diagonal_values[i]);	
	}

    float *dev_non_diagonal_values = 0;
	float *dev_diagonal_values = 0;
    int *dev_indeces = 0;
	float *dev_y = 0 ;
    float *dev_x = 0;

    cudaSetDevice(0);
	
    // Allocate GPU buffers
    cudaMalloc((void**)&dev_x, size * sizeof(float));
    cudaMalloc((void**)&dev_non_diagonal_values, 2 * size * sizeof(float));
    cudaMalloc((void**)&dev_diagonal_values, size * sizeof(float));
    cudaMalloc((void**)&dev_indeces, 2 * size * sizeof(int));
	cudaMalloc((void**)&dev_y, size * sizeof(float));
   
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_diagonal_values, diagonal_values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_non_diagonal_values, non_diagonal_values, 2 * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_indeces, indeces, 2 * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each row.
	jacobiFirstLocal<<<ceil(size/(1*32.0)), 1*32>>>(dev_x, dev_diagonal_values , dev_non_diagonal_values , dev_indeces , dev_y , size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	 cudaDeviceSynchronize();
	
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpyAsync(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	
	free(diagonal_values);
	free(non_diagonal_values) ;
	free(indeces);
	free(x);
	free(y);
	free(output) ;
	cudaFree(dev_x);
	cudaFree(dev_y);
    cudaFree(dev_diagonal_values);
    cudaFree(dev_non_diagonal_values);
    cudaFree(dev_indeces);
	cudaDeviceReset();
}

char* concat(char *s1, char *s2)
{
    char *result = (char *)malloc(strlen(s1)+strlen(s2)+1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}
