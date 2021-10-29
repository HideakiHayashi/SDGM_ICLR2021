%======================================
%calc_Hess_ww
%======================================
function Hess_ww = calc_Hess_ww(O2,Phi,usedWeightFlag,param)
Hess_ww = zeros(param.usedWeightNum,param.usedWeightNum);

pastUsedWeightNum1 = 0;
for c=1:param.classNum
   for m=1:param.maxCompNum
       
       if c~=param.classNum||m~=param.maxCompNum
           tmpUsed1 = usedWeightFlag(param.pastCompNum(c)*param.H+(m-1)*param.H+1:param.pastCompNum(c)*param.H+m*param.H);
           pastUsedWeightNum2 = 0;
           for cc=1:param.classNum
                for mm=1:param.maxCompNum
                    
                    if cc~=param.classNum||mm~=param.maxCompNum
                        tmpUsed2 = usedWeightFlag(param.pastCompNum(cc)*param.H+(mm-1)*param.H+1:param.pastCompNum(cc)*param.H+mm*param.H);
                        B = diag((delta(c,cc,m,mm)-O2(:,param.pastCompNum(cc)+mm)).*O2(:,param.pastCompNum(c)+m));
                        Hess_ww(pastUsedWeightNum2+1:pastUsedWeightNum2+sum(tmpUsed2),pastUsedWeightNum1+1:pastUsedWeightNum1+sum(tmpUsed1)) = ...
                                    Phi(:,tmpUsed2)'*B*Phi(:,tmpUsed1);
                        pastUsedWeightNum2 = pastUsedWeightNum2 + sum(tmpUsed2);
                    end
                    
                end
           end
           pastUsedWeightNum1 = pastUsedWeightNum1 + sum(tmpUsed1);
       end

    end
end

% for c=1:param.classNum-1
%    for m=1:param.maxCompNum
%        for cc=1:param.classNum-1
%             for mm=1:param.maxCompNum
%                 
%                 for h=1:param.H
%                     for hh=1:param.H
%                         Hess_ww((cc-1)*param.maxCompNum*param.H+(mm-1)*param.H+hh,(c-1)*param.maxCompNum*param.H+(m-1)*param.H+h) = ...
%                             (( -delta(c,cc,m,mm)+O2(:,(cc-1)*param.maxCompNum+mm) + (T(:,cc)./(Y(:,cc).^2)).*(delta(c,cc,m,mm).*Y(:,cc)-bigDelta(cc,c).*O2(:,(cc-1)*param.maxCompNum+mm)) ).*O2(:,(c-1)*param.maxCompNum+m).*Phi(:,h))'*Phi(:,hh);
%                     end
%                 end
%             end
%        end
% 
%     end
% end

