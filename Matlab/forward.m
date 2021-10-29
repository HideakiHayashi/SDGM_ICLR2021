%======================================
%forward calculation
%======================================
function [O2, Y] = forward(w, Phi, mixture, param,usedWeightFlag)
exI2 = zeros(param.N, param.tCompNum-1);    %
O2 = zeros(param.N, param.tCompNum);        %
Y = zeros(param.N, param.classNum);         %
sum = mixture(end,end).*ones(param.N, 1);   %

%
for c=1:param.classNum
    for m=1:param.maxCompNum
        if c~=param.classNum||m~=param.maxCompNum
%             exI2(:,param.pastCompNum(c)+m) = exp(Phi*w(param.pastCompNum(c)*param.H+(m-1)*param.H+1:param.pastCompNum(c)*param.H+m*param.H));
            tmp_w = w(param.pastCompNum(c)*param.H+(m-1)*param.H+1:param.pastCompNum(c)*param.H+m*param.H);
            tmpUsed = usedWeightFlag(param.pastCompNum(c)*param.H+(m-1)*param.H+1:param.pastCompNum(c)*param.H+m*param.H);
            exI2(:,param.pastCompNum(c)+m) = exp(Phi(:,tmpUsed)*tmp_w(tmpUsed));
            sum = sum + mixture(c,m).*exI2(:,param.pastCompNum(c)+m);
        end
    end
end
%
for c=1:param.classNum
    for m=1:param.maxCompNum
        if c==param.classNum&&m==param.maxCompNum
            O2(:,param.pastCompNum(c)+m) = mixture(c,m)./sum;
        else
            O2(:,param.pastCompNum(c)+m) = mixture(c,m).*exI2(:,param.pastCompNum(c)+m)./sum;
        end
            Y(:,c) = Y(:,c) + O2(:,param.pastCompNum(c)+m);
    end
end