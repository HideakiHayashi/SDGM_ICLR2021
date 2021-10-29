%======================================
%======================================
function [result] = prob2decision(prob,param)

I = eye(param.classNum);
[val ind] = max(prob,[],2);
result = I(ind,:);