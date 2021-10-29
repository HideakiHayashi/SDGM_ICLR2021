%======================================
%======================================
function [Phi] = calcGramMatrix(X,Y)

Phi = (X'*Y+1).^2;
N = size(Phi,1);
Phi = [Phi ones(N,1)];