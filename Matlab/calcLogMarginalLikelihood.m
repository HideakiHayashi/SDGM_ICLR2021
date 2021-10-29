%======================================
%======================================
function [J] = calcLogMarginalLikelihood(Y, T, param)
J = 0.0;                    %‘Î”ü•Ó‰»–Ş“x‚Ì‘æ1€
for c=1:param.classNum
%     for m=1:param.maxCompNum
%         logterm = log(O2(:,param.pastCompNum(c)+m));
        logterm = log(Y(:,c));
        logterm(logterm<-1e5) = -1e5;
%         J = J + (r(:,param.pastCompNum(c)+m).*T(:,c))'*logterm;
        J = J + T(:,c)'*logterm;
%     end
end