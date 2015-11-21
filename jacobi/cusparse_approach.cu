#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cusparse.h>

char* concat(char*,char*);

__global__ void divide(float *x, float* y ,float* out ,const int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		out[index] = x[index]/y[index] ;	
	}
}

void jacobi_cusparse(const int size , char* file_name)
{
	//initialize our test cases
   
	int nnz = 0 ;
	const float alpha = -1.0;
    const float beta = 1.0;
	float *values = (float *)malloc(2 * size * sizeof(float));
	float *diagonal_values = (float *)malloc(size * sizeof(float));
	int *rowPtr = (int *)malloc((size + 1) * sizeof(int));
	int *colIdx = (int *)malloc(2 * size * sizeof(int));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));

	char* diagonal_values_file_name = concat(file_name,"/csr_diagonal/csr_diagonal_value.txt") ;
	char* non_diagonal_values_file_name = concat(file_name,"/csr_diagonal/csr_non_diagonal_value.txt");
	char* rowPtr_file_name = concat(file_name,"/csr_diagonal/csr_rowPtr.txt");
	char* colIdx_file_name = concat(file_name,"/csr_diagonal/csr_colIdx.txt");
	char* y_file_name = concat(file_name,"/right_hand_side.txt");
	char* output_file_name = concat(file_name,"/output.txt");

	FILE *diagonal_values_file = fopen(diagonal_values_file_name, "r");
	FILE *non_diagonal_values_file = fopen(non_diagonal_values_file_name, "r");
	FILE *rowPtr_file = fopen(rowPtr_file_name, "r");
	FILE *colIdx_file = fopen(colIdx_file_name, "r");
	FILE *y_file = fopen(y_file_name, "r");
	FILE *output_file = fopen(output_file_name, "r");

	for (int i = 0 ; i < size ; i++)
	{
		fscanf(diagonal_values_file, "%f", &diagonal_values[i]);	
		fscanf(y_file, "%f", &y[i]);	
		fscanf(output_file, "%f", &output[i]);
		fscanf(rowPtr_file, "%d", &rowPtr[i]);
		x[i] = 0 ;
		if (i == size -1 )
		{
			fscanf(rowPtr_file, "%d", &rowPtr[i+1]);
		}
	}

	for (int i = 0 ; i< 2*size ; i++)
	{
		if (fscanf(colIdx_file, "%d", &colIdx[i]) == 1)
		{
			nnz++ ;
		}
		fscanf(non_diagonal_values_file, "%f", &values[i]);	
	}

	float *dev_values = 0 ;
	float *dev_diagonal_values = 0 ;
	int *dev_rowPtr = 0 ;
	int *dev_colIdx = 0 ;
	float *dev_x = 0 ;
	float *dev_y = 0 ;
 
	//Define the cusparse opaque structures
    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);

	cusparseMatDescr_t descr = 0;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

     //Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);
    
	 //Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&dev_x, size * sizeof(float));
	cudaMalloc((void**)&dev_y, size * sizeof(float));
	cudaMalloc((void**)&dev_values, nnz * sizeof(float));
	cudaMalloc((void**)&dev_diagonal_values, nnz * sizeof(float));
	cudaMalloc((void**)&dev_rowPtr, (size + 1) * sizeof(int));
	cudaMalloc((void**)&dev_colIdx, nnz * sizeof(int));
  
	//Memcpy
	cudaMemcpyAsync(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_diagonal_values, diagonal_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_rowPtr, rowPtr, (size + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_colIdx, colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 0 ; i<30 ; i++)
	{
		cudaMemcpyAsync(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);

		cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            size, size, nnz, &alpha,
                                            descr, dev_values, dev_rowPtr,
                                            dev_colIdx, dev_x, &beta,dev_y);
		divide<<<ceil(size/32.0),32>>>(dev_y,dev_diagonal_values,dev_x,size);	
	}

	cudaMemcpyAsync(x, dev_x, size*sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpyAsync(y, dev_y, size*sizeof(float), cudaMemcpyDeviceToHost );

	 /*cudaDeviceReset must be called before exiting in order for profiling and
     tracing tools such as Nsight and Visual Profiler to show complete traces.*/
    cudaDeviceReset();
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_values);
	cudaFree(dev_rowPtr);
	cudaFree(dev_colIdx);
}

