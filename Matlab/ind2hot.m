%======================================
%int2hot
%======================================
function [hot] = ind2hot(index, style)
if style=='posneg'
    index = (index+1)/2+1;
    I = eye(2,2);
    hot = I(index,:);
    
elseif style=='onestart'
    I = eye(max(index));
    hot = I(index,:);
end
    
    


