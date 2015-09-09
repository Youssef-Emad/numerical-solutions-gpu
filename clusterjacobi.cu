#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t jacobiCuda(const int cluserSize,float * oldRowNum,  float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces,const float *y, const int size);

__global__ void jacobiOne(const int clusterSize,float *oldRowNum,float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
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
}


__global__ void jacobiOneSharedAndLocal(const int clusterSize,float* oldRowNum,float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
	float local_diagonal_value ;
	float local_non_diagonal_values[2];
	float local_y;
	int oldIdx ;
	
	__shared__ float shared_x[4]; // shared_x[clusterSize + 2] ...... lma yossef yshof el dynamic allocation.
	

	local_diagonal_value = diagonal_values[index];
	local_non_diagonal_values[0] = non_diagonal_values[2 * index];
	local_non_diagonal_values[1] = non_diagonal_values[2 * index+1];
	local_y = y[index];
	shared_x[threadIdx.x + 1] = 0; // initialize the shared memory location as 0
	shared_x[0] = 0;//fill first and last positions with dummy values
	shared_x[3]=0;

	float sum = 0 ;
	if (threadIdx.x < clusterSize) // ensure you are withing the cluster
	{
		oldIdx = oldRowNum[index];
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
		x[oldIdx] = shared_x[threadIdx.x+1];
		
	}
	
}



int main()
{
	//initialize our test cases
    const int arraySize = 12;
	const int clusterSize =2;
	float oldRowNum[12] = {0,1,2,7,3,8,4,6,5,10,9,11};
	float non_diagonal_values[24] = {0,0.0185,0.0185,0,0,0.0185,0.0185,0,0,0.0185,0.0185,0,0,0.0185,0.0185,0,0,0.0185,0.0185,0,0,0.0185,0.0185,0} ;
	float diagonal_values[12] = {};
	int indeces[24] = {0,1,0,0,0,7,2,0,0,8,3,0,0,6,4,0,0,10,5,0,0,11,9,0};
    float x[arraySize] = { 0};
	float y[arraySize] = {};
	for (int i = 0 ; i<arraySize ; i++)
	{
		y[i] = 0.0878 ;
		diagonal_values[i] = 0.0741;
	}

	
    cudaError_t cudaStatus = jacobiCuda(clusterSize,oldRowNum,x,diagonal_values, non_diagonal_values, indeces,y, arraySize);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "JacobiCuda failed!");
        return 1;
    }

	printf("%f\n",x[0]);
	printf("%f\n",x[1]);
	printf("%f\n",x[2]);

	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaDeviceReset failed!");
        return 1;
    }
	system("pause");
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t jacobiCuda(const int clusterSize,float *oldRowNum, float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces,const float *y, const int size)
{
    float *dev_non_diagonal_values = 0;
	float *dev_diagonal_values = 0;
    int *dev_indeces = 0;
	float *dev_y = 0 ;
    float *dev_x = 0;
	float *dev_old = 0;
	


	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
	

    if (cudaStatus != cudaSuccess) {
		printf("%c\n", "cudaSetDevice failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMalloc for x failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_diagonal_values, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMalloc for diagonal values failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_non_diagonal_values, 2 * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMalloc for non diagonal values failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_indeces,  2 * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMalloc for indeces failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_y, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMalloc for y failed!");
        goto Error;
    }
	
	cudaStatus = cudaMalloc((void**)&dev_old,   size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMalloc for old num failed!");
        goto Error;
    }
    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_diagonal_values, diagonal_values, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMemcpy for diagonal values failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_non_diagonal_values, non_diagonal_values, 2 * size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMemcpy for non_diagonal values failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_indeces, indeces, 2 * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMemcpy for indeces failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMemcpy for y failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMemcpy for x failed!");
        goto Error;
    }
	
	cudaStatus = cudaMemcpy(dev_old, oldRowNum, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMemcpy for old num values failed!");
        goto Error;
    }
    // Launch a kernel on the GPU with one thread for each element.
	cudaEventRecord(start, 0);
	
	const dim3 blockDim(clusterSize,1,1);
	const dim3 gridDim(6,1,1);
    jacobiOneSharedAndLocal<<<gridDim, blockDim>>>(clusterSize,dev_old,dev_x, dev_diagonal_values , dev_non_diagonal_values , dev_indeces , dev_y , size);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
		printf("%c\n", "JacobiOne launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf( "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("%c\n", "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_x);
    cudaFree(dev_non_diagonal_values);
    cudaFree(dev_indeces);
	cudaFree(dev_old);
	/*cudaFree(dev_numCols);
	cudaFree(dev_numDiags);*/
    return cudaStatus;
}
