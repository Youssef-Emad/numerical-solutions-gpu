%first approach
[row , col , value] = find(matM);
size = size(row);
size = size(1);

result = [0 0 0] ;
for i = 1: size
    result(i,:) = [row(i) col(i) value(i)] ;
end

result = sortrows(result);

diagonal_count = 1;
non_diagonal_count = 1 ;
values_in_row_count = 0 ;
current_row_value = 1 ;
for j = 1:size
    if result(j,1) == result(j,2)
        diagonal(diagonal_count) = result(j,3) ;
        diagonal_count = diagonal_count + 1 ;
        
        if j == size && values_in_row_count < 2
            non_diagonal(non_diagonal_count) = 0 ;
            indeces(non_diagonal_count) = result(j-1,2) - 1 ;
            non_diagonal_count = non_diagonal_count + 1 ;
        end
        
    elseif result(j,1) ~= result(j,2) && result(j,1) == current_row_value && values_in_row_count < 2
        non_diagonal(non_diagonal_count) = result(j,3) ;
        indeces(non_diagonal_count) = result(j,2) -1 ;
        non_diagonal_count = non_diagonal_count + 1 ;
        current_row_value = result(j,1) ;
        values_in_row_count = values_in_row_count +1 ;
        
        if result(j+1,1) ~= result(j+1,2)
            next_to_check = result(j+1,1) ;
        elseif j+2 <= size
            next_to_check = result(j+2,1) ;
        end
        if values_in_row_count == 2
            values_in_row_count = 0 ;
            current_row_value = next_to_check;
        elseif next_to_check ~= current_row_value && values_in_row_count < 2
            for k = 1: 2 - values_in_row_count
                non_diagonal(non_diagonal_count) = 0 ;
                indeces(non_diagonal_count) = result(j,2) - 1 ;
                non_diagonal_count = non_diagonal_count + 1 ;
            end
            values_in_row_count = 0 ;
            current_row_value = next_to_check;
        end
    end
end