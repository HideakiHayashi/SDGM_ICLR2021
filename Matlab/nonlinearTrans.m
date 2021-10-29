%======================================
%Nonlinear transformation
%======================================
function [Phi] = nonlinearTrans(X)
[N D] = size(X);
H = 1+D*(D+3)/2;
Phi = zeros(N,H);
nonlinTerm = zeros(N,D*(D+1)/2);

Phi(:,1) = ones(N,1);
Phi(:,2:D+1) = X;

cnt=1;
for i=1:D
    for j=i:D
        nonlinTerm(:,cnt) = X(:,i).*X(:,j);
        cnt = cnt + 1;
    end
end

Phi(:,D+2:end) = nonlinTerm;
