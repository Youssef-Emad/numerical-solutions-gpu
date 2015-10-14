#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

__global__ void cg(float* a , int * indeces , float* b , float* x,int size) 
{
	int index = threadIdx.x ;
	__shared__  float r[1000] ;
	__shared__  float r_squared[1000] ;
	__shared__  float p_sum[1000] ;

	if (index < size)
	{
		float sum = 0 ;

		for (int i = 0 ; i<3 ; i++)
		{
			sum += a[i + 3*index] * x[indeces[i + 3*index]] ;
		}
		
		r[index] = b[index] - sum ;	
		p_sum[index] = 0 ;
		__syncthreads();

		for (int i = 0 ; i<3 ; i++)
		{
			p_sum[index] += a[i + 3*index] * r[indeces[i + 3*index]] ;
		}
		
		//calc alpha
		r_squared[index] = r[index] * r[index] ;
		p_sum[index] = p_sum[index] * r[index] ;

		for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
		{
			if (index < size/2)
			{
				// summation of r*rT
				r_squared[index] = r_squared[2*index] + r_squared[2*index + 1] ;
				//summation of r*a*rT
				p_sum[index] = p_sum[index] +  p_sum[index + s] ;
			}
			__syncthreads();
		}
	
		float alpha = r_squared[0]/p_sum[0] ;
		x[index] = x[index] + alpha * r[index] ;

	}

}

void main()
{
	//initialize our test cases

	const int size = 12 ;
	float a[3*size] = { 0.0741,0.0185,0, 0.0185,0.0741,0, 0.0741,0.0185,0, 0.0741,0.0185,0, 0.0741,0.0185,0 ,0.0741,0.0185,0, 0.0741,0.0185,0, 0.0741,0.0185,0, 0.0741,0.0185,0, 0.0741,0.0185,0, 0.0741,0.0185,0,0.0741,0.0185,0} ;
	int indeces[3*size] = {0,1,1,0,1,1,2,7,7,3,8,8,4,6,6,5,10,10,6,4,4,7,2,2,8,3,3,9,11,11,10,5,5,11,9,9} ;
	float b[size] = {} ;
	float x[size] = {};
	
	for (int i = 0 ; i < size ; i++)
	{
		b[i] = 0.0878 ;
		x[i] = 0.0878 ;
	}
	float* dev_a = 0;
	int* dev_indeces = 0 ;
	float* dev_b = 0;
	float* dev_x = 0;

    cudaSetDevice(0);
	
    // Allocate GPU buffers
   
    cudaMalloc((void**)&dev_a, 3 * size * sizeof(float));
	cudaMalloc((void**)&dev_indeces, 3 * size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(float));
    cudaMalloc((void**)&dev_x, size * sizeof(float));
   
    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyAsync(dev_a, a, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_indeces, indeces, 3 * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each row.
	cg<<<1,size>>>(dev_a,dev_indeces,dev_b,dev_x,size);
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpyAsync(x, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("%f\n",x[0]);
	printf("%f\n",x[1]);
	printf("%f\n",x[11]);
	cudaDeviceReset();
	system("pause");
}


