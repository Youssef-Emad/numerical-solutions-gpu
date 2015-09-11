//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <cusparse.h>
//
//void cusparse_solver()
//{
//	//initialize our test cases
//    const int m = 4;
//	const int nnz = 4 ;
//	const float alpha = 1.0;
//
//	float values[] = {0,0,0,0} ;
//	float diagonal_values[] = {0.1667,0.1667,0.1667,0.1667} ;
//	int rowptr[] = {0,1,2,3,4,7};
//	int colidx[] = {0,1,2,3};
//
//	float y[] = {0.2036,0.2036,0.2036,0.2036};
//	float x[4] ;
//
//	float *dev_values = 0 ;
//	float *dev_diagonal_values = 0 ;
//	int *dev_rowptr = 0 ;
//	int *dev_colidx = 0 ;
//	float *dev_x = 0 ;
//	float *dev_y = 0 ;
// 
//	//define the cusparse opaque structures
//    cusparsehandle_t handle;
//    cusparsecreate(&handle);
//	cusparsesolveanalysisinfo_t info = 0;
//	cusparsecreatesolveanalysisinfo(&info);
//	cusparsematdescr_t descr = 0;
//
//    cusparsecreatematdescr(&descr);
//    cusparsesetmattype(descr,cusparse_matrix_type_general);
//    cusparsesetmatindexbase(descr,cusparse_index_base_zero);
//
//    // choose which gpu to run on, change this on a multi-gpu system.
//    cudasetdevice(0);
//
//	// allocate gpu buffers for three vectors (two input, one output)    .
//    cudamalloc((void**)&dev_x, m * sizeof(float));
//	cudamalloc((void**)&dev_y, m * sizeof(float));
//	cudamalloc((void**)&dev_values, nnz * sizeof(float));
//	cudamalloc((void**)&dev_diagonal_values, nnz * sizeof(float));
//	cudamalloc((void**)&dev_rowptr, (m+1) * sizeof(int));
//	cudamalloc((void**)&dev_colidx, nnz * sizeof(int));
//  
//	//memcpy
//	cudamemcpyasync(dev_x, x, m * sizeof(float), cudamemcpyhosttodevice);
//	cudamemcpyasync(dev_values, values, nnz * sizeof(float), cudamemcpyhosttodevice);
//	cudamemcpyasync(dev_diagonal_values, diagonal_values, nnz * sizeof(float), cudamemcpyhosttodevice);
//	cudamemcpyasync(dev_rowptr, rowptr, (m+1) * sizeof(int), cudamemcpyhosttodevice);
//	cudamemcpyasync(dev_colidx, colidx, nnz * sizeof(int), cudamemcpyhosttodevice);
//	cudamemcpyasync(dev_y, y, m * sizeof(float), cudamemcpyhosttodevice);
//
//	cusparsescsrsv_analysis(handle, cusparse_operation_non_transpose, m, m, descr, dev_diagonal_values, dev_rowptr, dev_colidx, info);
//	cusparsescsrsv_solve(handle, cusparse_operation_non_transpose, m, &alpha, descr, dev_diagonal_values, dev_rowptr, dev_colidx, info, dev_y, dev_x);
//	
//	cudamemcpyasync(x, dev_x, m*sizeof(float), cudamemcpydevicetohost );
//
//
//	// cudadevicereset must be called before exiting in order for profiling and
//    // tracing tools such as nsight and visual profiler to show complete traces.
//    cudadevicereset();
//   	cusparsedestroysolveanalysisinfo(info);
//	cusparsedestroy(handle);
//    cudafree(dev_x);
//    cudafree(dev_y);
//    cudafree(dev_values);
//	cudafree(dev_rowptr);
//	cudafree(dev_colidx);
//}
//
