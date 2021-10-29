%======================================
%Calculate the posterior probability of mixture
%======================================
function [r] = calcPostMix(O2, Y, param)
r = zeros(param.N, param.tCompNum);

for c=1:param.classNum
    for m=1:param.maxCompNum
        r(:,param.pastCompNum(c)+m) = O2(:,param.pastCompNum(c)+m)./Y(:,c);
    end
end