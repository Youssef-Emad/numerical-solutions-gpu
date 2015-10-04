#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

char* concat(char*,char*);
/*__global__ void jacobiClustredGlobal(const int clusterSize,float *oldRowNum,float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x; 
	float sum = 0 ;
	int oldIdx; // the old row index of the row being processed in thread
	if (index < size)
	{
		oldIdx = oldRowNum[index];
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = 0 ; i<2 ; i++)
			{
				sum += non_diagonal_values[ i +2 * index]  * x[indeces[i  +2 * index]] ;
			}
			x[oldIdx] = (y[oldIdx] - sum )/diagonal_values[oldIdx];
			sum = 0 ;
			__syncthreads();	
		}
	}
}*/


__global__ void jacobiClusteredLocal(const int clusterSize,float *x, const float *diagonal_values , const float *non_diagonal_values,const float *y, const int size)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
	float local_diagonal_value ;
	float local_non_diagonal_values[2];
	float local_y;
	
	
	extern __shared__ float shared_x[];

	local_diagonal_value = diagonal_values[index];
	local_non_diagonal_values[0] = non_diagonal_values[2 * index];
	local_non_diagonal_values[1] = non_diagonal_values[2 * index+1];
	local_y = y[index];
	shared_x[threadIdx.x + 1] = 0; // initialize the shared memory location as 0
	shared_x[0] = 0;//fill first and last positions with dummy values
	shared_x[clusterSize + 1] = 0;

	float sum = 0 ;
	if (threadIdx.x < clusterSize) // ensure you are withing the cluster
	{
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = 0 ; i<2 ; i++)
			{
				sum += local_non_diagonal_values[i]  * shared_x[i*2 + threadIdx.x ] ; //all memory access is in shared memory
			}
			shared_x[threadIdx.x+1] = (local_y - sum )/local_diagonal_value;
			sum = 0 ;
			__syncthreads();	
		}
		x[index] = shared_x[threadIdx.x+1];
	}
}



void jacobiCuda(const int noElem , char* file_name)
{
    //initialize our test cases
	const int size = (2 * noElem)*(noElem -1);
	float *non_diagonal_values = (float *)malloc(2 * size * sizeof(float));
	float *diagonal_values = (float *)malloc(size * sizeof(float));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));
	
	const int clusterSize =noElem - 1;
	
	char* diagonal_values_file_name = concat(file_name,"/Diag.txt") ;
	char* non_diagonal_values_file_name = concat(file_name,"/values.txt");
	char* y_file_name = concat(file_name,"/B.txt");
	char* output_file_name = concat(file_name,"/out.txt");
	
	FILE *diagonal_values_file = fopen(diagonal_values_file_name, "r");
	FILE *non_diagonal_values_file = fopen(non_diagonal_values_file_name, "r");
	FILE *y_file = fopen(y_file_name, "r");
	FILE *output_file = fopen(output_file_name, "w");

	for (int i = 0 ; i < size ; i++)
	{
		fscanf(diagonal_values_file, "%f", &diagonal_values[i]);	
		fscanf(y_file, "%f", &y[i]);	
		x[i] = 0 ;
	}

	for (int i = 0 ; i< 2*size ; i++)
	{
		fscanf(non_diagonal_values_file, "%f", &non_diagonal_values[i]);	
	}

	float *dev_non_diagonal_values = 0;
	float *dev_diagonal_values = 0;
    float *dev_y = 0 ;
    float *dev_x = 0;
	
    // Choose which GPU to run on, change this on a multi-GPU system.
	cudaSetDevice(0);
    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_x, size * sizeof(float));
   	cudaMalloc((void**)&dev_diagonal_values, size * sizeof(float));
    cudaMalloc((void**)&dev_non_diagonal_values, 2 * size * sizeof(float));
    cudaMalloc((void**)&dev_y, size * sizeof(float));
	
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_diagonal_values, diagonal_values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_non_diagonal_values, non_diagonal_values, 2 * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    // Launch a kernel on the GPU with one thread for each element.
	const dim3 blockDim(clusterSize,1,1);
	const dim3 gridDim(2 * noElem,1,1);
    jacobiClusteredLocal<<<gridDim, blockDim,(clusterSize + 2) * sizeof(float)>>>(clusterSize,dev_x, dev_diagonal_values , dev_non_diagonal_values , dev_y , size);
	//cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    
    // Copy output vector from GPU buffer to host memory.
   cudaMemcpyAsync(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
   
   printf("%f",x[0]);
   printf("%f",x[1]);
   printf("%f",x[2]);
   for (int i = 0 ; i< size ; i++)
	{
		fprintf(output_file, "%f", &x[i]);		//ouput always 0.000000	
		
	}
	
	free(diagonal_values);
	free(non_diagonal_values) ;
	free(x);
	free(y);
	free(output) ;
    cudaFree(dev_x);
    cudaFree(dev_non_diagonal_values);
    cudaDeviceReset();
    
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

	
	jacobiCuda(15,"D:\codetest");


	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    system("pause");
    return 0;
}
