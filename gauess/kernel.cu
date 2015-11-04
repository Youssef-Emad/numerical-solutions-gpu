#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

char* concat(char*,char*);

// Kernel
__global__ void GaussianEliminationGlobal(const int clusterSize,float *x, const float *diagonal_values , const float *non_diagonal_values ,float *y , const int size)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x ;
	const int gi = index * clusterSize;

	float matrix[180][180];	//size of matrix
	for (int i = gi; i < gi + clusterSize;++i)
	{
		for (int j = gi; j < gi + clusterSize;++j)
		{
			matrix[i][j]=0;
			
		}
		matrix[i][i] = diagonal_values[i];
	}
	for(int i = gi; i < gi + clusterSize - 1 ;++i)
	{
		matrix[i][i+1] = non_diagonal_values[2*i+1];
		matrix[i+1][i] = non_diagonal_values[2*i+2];
	}

	// triangle form
	for (int i = gi ; i < gi + clusterSize; ++i)
   {
        //for every row...
        for (int j = i+1; j < gi + clusterSize; ++j)
        {
            //calculate ratio for every row below it using the triangular
            double ratio = matrix[j][i] / matrix[i][i];
            for(int k = gi; k < gi + clusterSize; ++k)
            {
                //Eliminate every column based on that ratio
                matrix[j][k] = matrix[j][k] - (matrix[i][k] * ratio);
            }
            //elimination on the coefficient vector
            y[j] = y[j] - (y[i] * ratio);
        }
	}
	__syncthreads();
//Back substitution
	for (int i = gi + clusterSize-1; i > gi-1; --i) 
    {
        double current = 0;
        for (unsigned int j = i; j < gi + clusterSize; ++j)
        {
            current = current + (matrix[i][j] * x[j]);
        }
        x[i] = (y[i] - current) / matrix[i][i];
    }
}


__global__ void GaussianEliminationShared(const int clusterSize,float *x, const float *diagonal_values , const float *non_diagonal_values ,float *y )
{
	const int index = blockIdx.x ;

	__shared__ float shared_m[9][9]; // size of cluster
	for (int i = 0; i < clusterSize;++i)
	{
		for (int j = 0; j < clusterSize;++j)
		{
			shared_m[i][j]=0;
		}
	}
	for(int i = 0; i < clusterSize; ++i)
	{
		shared_m[i][i] = diagonal_values[clusterSize * index + i];
	}
	for(int i = 0; i < clusterSize-1;++i)
	{
		shared_m[i][i+1] = non_diagonal_values[clusterSize * index * 2 + 2*i+1];
		shared_m[i+1][i] = non_diagonal_values[clusterSize * index * 2 + 2*i+2];
	}

	// triangle form
	for (int i = 0 ; i < clusterSize; ++i)
   {
        //for every row...
        for (int j = i+1; j < clusterSize; ++j)
        {
            //calculate ratio for every row below it using the triangular
            double ratio = shared_m[j][i] / shared_m[i][i];
            for(int k = 0; k < clusterSize; ++k)
            {
                //Eliminate every column based on that ratio
                shared_m[j][k] = shared_m[j][k] - (shared_m[i][k] * ratio);
            }
            //elimination on the coefficient vector
            y[clusterSize * index +j] = y[clusterSize * index +j] - (y[clusterSize * index +i] * ratio);
        }
	}
	__syncthreads();

	//Back substitution
	for (int i = clusterSize-1; i > -1; --i)
    {
        double current = 0;
        for (unsigned int j = i; j < clusterSize; ++j)
        {
            current = current + (shared_m[i][j] * x[clusterSize * index +j]);
        }
        x[clusterSize * index +i] = (y[clusterSize * index +i] - current) / shared_m[i][i];
    }
	
}

void gaussianCuda(const int noElem, char* file_name)
{
	const int size = (2 * noElem)*(noElem -1);
	float *non_diagonal_values = (float *)malloc(2 * size * sizeof(float));
	float *diagonal_values = (float *)malloc(size * sizeof(float));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));
	
	const int clusterSize = noElem - 1;
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
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
	for (int i = 0 ; i< 2*size; i++)
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
	const dim3 blockDim(noElem,1,1);
	const dim3 gridDim(2 ,1,1);

	cudaEventRecord(start, 0);
	//uncomment the methode you want to use
    //GaussianEliminationGlobal<<<gridDim, blockDim/*,(clusterSize * clusterSize) * sizeof(float)*/>>>(clusterSize,dev_x, dev_diagonal_values , dev_non_diagonal_values , dev_y,size);
	//GaussianEliminationShared<<<2*noElem, 1/*,(clusterSize * clusterSize) * sizeof(float)*/>>>(clusterSize,dev_x, dev_diagonal_values , dev_non_diagonal_values , dev_y);
		
	//cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);
    // Copy output vector from GPU buffer to host memory.
   cudaMemcpyAsync(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	// Write the values in X array to the file out
   for (int i = 0 ; i< size ; i++)
	{
		fprintf(output_file, "%f",x[i]);			
	}
	
	fclose(output_file);
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
	gaussianCuda(10,"D:\codetest");
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    system("pause");
    return 0;
}