%csr
index = 1;
for i = 1:12
    for j = 1:12 
       if matM(i,j) ~= 0 
        value(index) = matM(i,j);
        rowPtr(i) = index;
        colIdx(index) = j;
        index = index + 1;
       end
    end
end

%first
index2 = 1;
count = 0 ;
for i = 1:12
    for j = 1:12 
       if matM(i,j) ~= 0 
        value2(index2) = matM(i,j);
        colIdx2(index2) = j;
        index2 = index2 + 1;
        count = count + 1 ;
       end
       if count == 1 
        value2(index2) = 0;
        colIdx2(index2) = 0;
        index2 = index2 + 1;
       end
       count =  0 ;
    end
end