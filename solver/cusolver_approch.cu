#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cusolverSp.h>
#include <assert.h>

char* concat(char *s1, char *s2)
{
    char *result = (char *)malloc(strlen(s1)+strlen(s2)+1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void cusparse_approach(int size ,char* file_name)
{
	int nnz =  0 ;
	int singularity = 0 ;

   //initialize our test cases
    float *values = (float *)malloc(3 * size * sizeof(float));
	int *rowPtr = (int *)malloc((size + 1) * sizeof(int));
	int *colIdx = (int *)malloc(3 * size * sizeof(int));
	float *x = (float *)malloc(size * sizeof(float));
	float *y = (float *)malloc(size * sizeof(float));

	char* values_file_name = concat(file_name,"/csr/csr_value.txt") ;
	char* rowPtr_file_name = concat(file_name,"/csr/csr_rowPtr.txt");
	char* colIdx_file_name = concat(file_name,"/csr/csr_colIdx.txt");
	char* y_file_name = concat(file_name,"/right_hand_side.txt");

	FILE *values_file = fopen(values_file_name, "r");
	FILE *rowPtr_file = fopen(rowPtr_file_name, "r");
	FILE *colIdx_file = fopen(colIdx_file_name, "r");
	FILE *y_file = fopen(y_file_name, "r");

	for (int i = 0 ; i < size ; i++)
	{
		fscanf(y_file, "%f", &y[i]);	
		fscanf(rowPtr_file, "%d", &rowPtr[i]);
		x[i] = 0 ;
		if (i == size -1 )		{
			fscanf(rowPtr_file, "%d", &rowPtr[i+1]);
		}
	}

	for (int i = 0 ; i< 3*size ; i++)
	{
		fscanf(colIdx_file, "%d", &colIdx[i]);
		fscanf(values_file, "%f", &values[i]);
		if(values[i] > 0)
		{
			nnz++ ;
		}
	}

	
    cusolverSpHandle_t solver_handle ;
    cusolverSpCreate(&solver_handle) ;
    cusparseMatDescr_t descr = 0;

    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
	
    cusolverSpScsrlsvluHost(solver_handle, size, nnz, descr, values, rowPtr, colIdx, y, 0.0,0, x, &singularity);
}