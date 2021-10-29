%======================================
%======================================
function [error] = calcClassificationError(Y,T)
N = size(T,1);
[val result] = max(Y,[],2);
[val label] = max(T,[],2);
error = sum(result~=label)./N;