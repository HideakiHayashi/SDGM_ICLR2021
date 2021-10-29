%======================================
%intsave
%======================================
function [] = indsave(filename, index)
    dlmwrite(filename,index,'delimiter','\t','precision',1);
    


