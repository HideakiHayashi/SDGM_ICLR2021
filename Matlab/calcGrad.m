%======================================
%calcGrad
%======================================
function [J_w] = calcGrad(T,r,O2,Phi,usedWeightFlag,param)
J_w = zeros(param.usedWeightNum,1);

pastUsedWeightNum = 0;
for c=1:param.classNum
    for m=1:param.maxCompNum
        if c~=param.classNum||m~=param.maxCompNum
            tmpUsed = usedWeightFlag(param.pastCompNum(c)*param.H+(m-1)*param.H+1:param.pastCompNum(c)*param.H+m*param.H);
            J_w(pastUsedWeightNum+1:pastUsedWeightNum+sum(tmpUsed)) = Phi(:,tmpUsed)'*(r(:,param.pastCompNum(c)+m).*T(:,c)-O2(:,param.pastCompNum(c)+m));
            pastUsedWeightNum = pastUsedWeightNum + sum(tmpUsed);
        end
    end
end

% for c=1:param.classNum-1
%     for m=1:param.maxCompNum
%         for h=1:param.H
%             J_w((c-1)*param.maxCompNum*param.H+(m-1)*param.H+h) = ((T(:,c)./Y(:,c)-1.0).*O2(:,(c-1)*param.maxCompNum+m))'*Phi(:,h);
%         end
% 
%     end
% end