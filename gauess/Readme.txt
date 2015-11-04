main function should only include a call for the void gaussianCuda(const int noElem, char* file_name) function.
this function takes the number of elements forming the matrix and the distnation of the folder containing the files to be read.
the cluster methode uses the following formulas to calculate the size of the cluster and their number.
noOfClusters = 2 * noElem;
clusterSize = noElem - 1;
the size of the matrix is noOfClusters * clusterSize.
you should uncomment the line of the kernel you ant to use.

future work
allocation of shared memory should be dynamic using the extern function.
make the triangle form loop only over nonzero elements in the matrix.  