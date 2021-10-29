%======================================

%======================================
function [mergedMixture,usedCompFlag] = mergeMixture(mixture,usedWeightFlag,param)

mergedMixture = mixture;
usedCompFlag = true(param.classNum,param.maxCompNum);
usedCompFlag(end,end) = false;
for c=1:param.classNum
    for m=1:param.maxCompNum
        if c~=param.classNum||m~=param.maxCompNum
            tmpUsed = usedWeightFlag(param.pastCompNum(c)*param.H+(m-1)*param.H+1:param.pastCompNum(c)*param.H+m*param.H);
            if sum(tmpUsed) == 0
                usedCompFlag(c,m) = false;
            end    
        end
    end
    mixtureNotUsed = mergedMixture(c,~usedCompFlag(c,:));
    if ~isempty(mixtureNotUsed)
        mixtureNotUsedSum = sum(mixtureNotUsed);
        mixtureNotUsed = zeros(size(mixtureNotUsed));
        mixtureNotUsed(1) = mixtureNotUsedSum;
        mergedMixture(c,~usedCompFlag(c,:)) = mixtureNotUsed;
    end
end
