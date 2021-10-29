%======================================
%intsave
%======================================
function [] = ind2hot_save(filename, index, style)
hot = ind2hot(index,style);
dlmwrite(filename,hot,'delimiter','\t','precision',1);
    


