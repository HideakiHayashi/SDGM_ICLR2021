%======================================
%Confirm internal output
%======================================

function [result] = confirm_internal_output(net,inTestDataName,inTestLabelName)

%�����`�F�b�N
if nargin ~= 3
    disp('Not enough arguments!');
    result = [];
    return
end

%=============
%�f�[�^���[�h
%=============
TestDataName = inTestDataName;       %�e�X�g�f�[�^�t�@�C����
TestLabelName = inTestLabelName;     %����
X_test = load(TestDataName);        %�e�X�g�f�[�^
X_test = X_test';
T_test = load(TestLabelName);       %����

%=============
%�p�����[�^�ǂݍ���
%=============
paramClassify = net.param;
paramClassify.N = size(X_test,2);   %�e�X�g�f�[�^��

%=============
%�O�����s��쐬
%=============
if net.iskernel
    Phi_test = calcGramMatrix(X_test,net.data.X);   %�J�[�l���g�p��
else
    Phi_test = nonlinearTrans(X_test');             %�Œ���g�p��
end

%=============
%����
%=============
[I2_test, exI2_test, O2_test, Y_test] = forward_detail(net.w, Phi_test, net.mixture, paramClassify, net.usedWeightFlag);

%=============
%���ʌ덷
%=============
TestingError = calcClassificationError(Y_test,T_test);
display(TestingError);

%=============
%�߂�l�܂Ƃ�
%=============
result = struct('TestingError',TestingError,'Y_test',Y_test,'O2_test',O2_test,'I2_test',I2_test,'exI2_test',exI2_test,'Phi_test',Phi_test,'paramClassify',paramClassify);
