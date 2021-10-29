%======================================
%Confirm internal output
%======================================

function [result] = confirm_internal_output(net,inTestDataName,inTestLabelName)

%引数チェック
if nargin ~= 3
    disp('Not enough arguments!');
    result = [];
    return
end

%=============
%データロード
%=============
TestDataName = inTestDataName;       %テストデータファイル名
TestLabelName = inTestLabelName;     %答え
X_test = load(TestDataName);        %テストデータ
X_test = X_test';
T_test = load(TestLabelName);       %答え

%=============
%パラメータ読み込み
%=============
paramClassify = net.param;
paramClassify.N = size(X_test,2);   %テストデータ数

%=============
%グラム行列作成
%=============
if net.iskernel
    Phi_test = calcGramMatrix(X_test,net.data.X);   %カーネル使用時
else
    Phi_test = nonlinearTrans(X_test');             %固定基底使用時
end

%=============
%識別
%=============
[I2_test, exI2_test, O2_test, Y_test] = forward_detail(net.w, Phi_test, net.mixture, paramClassify, net.usedWeightFlag);

%=============
%識別誤差
%=============
TestingError = calcClassificationError(Y_test,T_test);
display(TestingError);

%=============
%戻り値まとめ
%=============
result = struct('TestingError',TestingError,'Y_test',Y_test,'O2_test',O2_test,'I2_test',I2_test,'exI2_test',exI2_test,'Phi_test',Phi_test,'paramClassify',paramClassify);
