%csr
csr_index = 1;
csr_rowPtr(1) = 0 ;
csr_value = 1:10;
for i = 1:1
    for j = 1:1 
       if matM(i,j) ~= 0 
        csr_value(csr_index) = matM(i,j);
        csr_rowPtr(i+1) = csr_index;
        csr_colIdx(csr_index) = j - 1;
        csr_index = csr_index + 1;
       end
    end
end

csr_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\csr_value.txt','w');
fprintf(csr_fileID,'%f\n',csr_value);
fclose(csr_fileID);

csr_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\csr_rowPtr.txt','w');
fprintf(csr_fileID,'%d\n',csr_rowPtr);
fclose(csr_fileID);

csr_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\csr_colIdx.txt','w');
fprintf(csr_fileID,'%d\n',csr_colIdx);
fclose(csr_fileID);

%csr diagonal
csr_index = 1;
csr_diagonal_rowPtr(1) = 0 ;
csr_non_diagonal_value = 1:10;
csr_diagonal_value = 1:10 ;
csr_diagonal_colIdx(1) = 0;
for i = 1:1
    for j = 1:1 
       if matM(i,j) ~= 0 && i ~= j
        csr_non_diagonal_value(csr_index) = matM(i,j);
        csr_diagonal_rowPtr(i+1) = csr_index;
        csr_diagonal_colIdx(csr_index) = j - 1;
        csr_index = csr_index + 1;
       end
       if i == j
           csr_diagonal_value(i) = matM(i,i) ;
       end
    end
end

csr_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\csr_non_diagonal_value.txt','w');
fprintf(csr_fileID,'%f\n',csr_non_diagonal_value);
fclose(csr_fileID);

csr_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\csr_diagonal_value.txt','w');
fprintf(csr_fileID,'%f\n',csr_diagonal_value);
fclose(csr_fileID);

csr_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\csr_rowPtr.txt','w');
fprintf(csr_fileID,'%d\n',csr_diagonal_rowPtr);
fclose(csr_fileID);

csr_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\csr_colIdx.txt','w');
fprintf(csr_fileID,'%d\n',csr_diagonal_colIdx);
fclose(csr_fileID);


%first
first_approach_index = 1;
first_approach_non_diagonal_values = 1 ;
first_appraoch_diagonal_values = 1 ;
first_approach_diagonal_values = 1 ;
count = 0 ;
for i = 1:24
    for j = 1:24
       if matM(i,j) ~= 0 && i~= j
        first_approach_non_diagonal_values(first_approach_index) = matM(i,j);
        first_approach_indeces(first_approach_index) = j - 1;
        first_approach_index = first_approach_index + 1;
        count = count + 1 ;
       end
       if i==j
           first_approach_diagonal_values(i) = matM(i,j) ;
       end
    end
    if count == 1 
        first_approach_non_diagonal_values(first_approach_index) = 0;
        first_approach_indeces(first_approach_index) = first_approach_indeces(first_approach_index - 1);
        first_approach_index = first_approach_index + 1;
    end
    count =  0 ;
end

first_approach_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\first_approach_non_diagonal_values.txt','w');
fprintf(first_approach_fileID,'%f\n',first_approach_non_diagonal_values);
fclose(first_approach_fileID);

first_approach_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\first_approach_diagonal_values.txt','w');
fprintf(first_approach_fileID,'%f\n',first_approach_diagonal_values);
fclose(first_approach_fileID);

first_approach_fileID = fopen('C:\Users\youssef\Desktop\numerical-solutions-gpu\jacobi\jacobi\first_approach_indeces.txt','w');
fprintf(first_approach_fileID,'%d\n',first_approach_indeces);
fclose(first_approach_fileID);

