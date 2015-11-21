#include <stdio.h>

void jacobiFirst(const int size , char* file_name);
void jacobiCsrOne(const int  size , char* file_name);
void jacobi_cusparse(const int size , char* file_name);

int main()
{
	jacobiFirst(420,"C:/Users/youssef/Desktop/numerical-solutions-gpu/jacobi/jacobi/test_cases/420");
	jacobiCsrOne(420,"C:/Users/youssef/Desktop/numerical-solutions-gpu/jacobi/jacobi/test_cases/420");
	jacobi_cusparse(420,"C:/Users/youssef/Desktop/numerical-solutions-gpu/jacobi/jacobi/test_cases/420");
	system("pause");
	return 1;
}