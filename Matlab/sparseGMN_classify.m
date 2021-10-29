%======================================
%Classification using Sparse GMN
%======================================

function [result] = sparseGMN_classify(net,inTestDataName,inTestLabelName)

%引数チェック
if nargin ~= 3
    disp('Not enough arguments!');
    result = [];
    return
end

%=============
%Load
%=============
TestDataName = inTestDataName;       
TestLabelName = inTestLabelName;     
X_test = load(TestDataName);        %
X_test = X_test';
T_test = load(TestLabelName);       %

%=============
%Parameters
%=============
paramClassify = net.param;
paramClassify.N = size(X_test,2);   %

%=============
%Gram mat
%=============
if net.iskernel
    Phi_test = calcGramMatrix(X_test,net.data.X);   
else
    Phi_test = nonlinearTrans(X_test');             
end

%=============

%=============
[O2_test, Y_test] = forward(net.w, Phi_test, net.mixture, paramClassify, net.usedWeightFlag);

%=============

%=============
TestingError = calcClassificationError(Y_test,T_test);
display(TestingError);

%=============

%=============
result = struct('TestingError',TestingError,'Y_test',Y_test,'O2_test',O2_test,'Phi_test',Phi_test,'paramClassify',paramClassify);
