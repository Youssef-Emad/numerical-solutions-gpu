#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void jacobiFirst();

__global__ void jacobiCsr(float *x, const float *diagonal_values , const float * values, const int *rowPtr ,const int *colIdx,const float *y, const int size)
{
    const int index = threadIdx.x;
	int row_start = rowPtr[index];
	int row_end = rowPtr[index +1];

	float sum = 0 ;

	if (index < size)
	{
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = row_start ; i< row_end ; i++)
			{
				sum += values[i] * x[colIdx[i]];
			}
			x[index] =( y[index] - sum )/diagonal_values[index] ;
			sum = 0 ;
			__syncthreads();	
		}
	}
}

__global__ void jacobiCsrShared(float *x, const float *diagonal_values , const float * values, const int *rowPtr ,const int *colIdx,const float *y, const int size)
{
    const int index = threadIdx.x;
	__shared__ float shared_diagonal_values[24] ;
	__shared__ float shared_values[48];
	__shared__ int shared_colIdx[24];
	__shared__ float shared_x[24];
	__shared__ float shared_y[24];

	shared_diagonal_values[index] = diagonal_values[index];
	shared_values[index] = values[index];
	shared_colIdx[index] = colIdx[index];
	shared_y[index] = y[index];
	shared_x[index] = x[index];

	int row_start = rowPtr[index];
	int row_end = rowPtr[index +1];

	float sum = 0 ;
	if (index < size)
	{
		for (int j = 0 ; j< 30 ; j++)
		{
			for (int i = row_start ; i< row_end ; i++)
			{
				sum += shared_values[i] * shared_x[shared_colIdx[i]];
			}
			shared_x[index] =(shared_y[index] - sum )/shared_diagonal_values[index] ;
			sum = 0 ;
			__syncthreads();	
		}
		x[index] = shared_x[index];
	}
}


void jacobiCsrOne()
{
	//initialize our test cases
	const int size = 4 ;
	float values[] = {0,0,0,0} ;
	float diagonal_values[] = {0.1667,0.1667,0.1667,0.1667} ;
	int rowPtr[] = {0,1,2,3,4,7};
	int colIdx[] = {0,1,2,3};

	float y[] = {0.2036,0.2036,0.2036,0.2036};
	float x[] = {0,0,0,0};

	float *dev_values = 0 ;
	float *dev_diagonal_values = 0 ;
	int *dev_rowPtr = 0 ;
	int *dev_colIdx = 0 ;
	float *dev_x = 0 ;
	float *dev_y = 0 ;

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaSetDevice(0);
	
    // Allocate GPU buffers
    cudaMalloc((void**)&dev_x, size * sizeof(float));
    cudaMalloc((void**)&dev_diagonal_values, size * sizeof(float));
    cudaMalloc((void**)&dev_values,  size * sizeof(int));
	cudaMalloc((void**)&dev_rowPtr,  (size+1) * sizeof(int));
	cudaMalloc((void**)&dev_colIdx,  size * sizeof(int));
	cudaMalloc((void**)&dev_y, size * sizeof(float));
   
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_diagonal_values, diagonal_values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_values, values,  size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_colIdx, colIdx,  size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_rowPtr, rowPtr, ( size+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    
	cudaEventRecord(start, 0);
    // Launch a kernel on the GPU with one thread for each element.
	jacobiCsrShared<<<1, size>>>(dev_x, dev_diagonal_values , dev_values , dev_rowPtr,dev_colIdx , dev_y , size);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	 cudaDeviceSynchronize();
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_x);
	cudaFree(dev_y);
    cudaFree(dev_diagonal_values);
    cudaFree(dev_values);
    cudaFree(dev_rowPtr);
	cudaDeviceReset();
}

int main() 
{
	jacobiCsrOne();
	//jacobiFirst();
	system("pause");
	return 1 ;
}