%======================================
%Demo for Repley's data classification using SDGM
%======================================
close all;
clear;
clc;

inTrainDataName = 'synthTrainData.dat';       
inTrainLabelName = 'synthTrainLabels.dat';    
inClassNum = 2;                     
inMaxCompNum = 3;                   
iskernel = true;

inTestDataName = 'synthTestData.dat';
inTestLabelName = 'synthTestLabels.dat';

net = sparseGMN_train(inTrainDataName,inTrainLabelName,inClassNum,inMaxCompNum,iskernel);
result = sparseGMN_classify(net,inTestDataName,inTestLabelName);
detail = confirm_internal_output(net,inTestDataName,inTestLabelName);
plot(detail.I2_test);
plot(detail.exI2_test);
semilogy(detail.exI2_test);
plot(detail.O2_test);
plot(detail.Y_test);

plotDecisionBoundary(net.w,net.mixture,net.data.X,net.data.T,net.usedWeightFlag,net.param);
[mergedMixture,usedCompFlag] = mergeMixture(net.mixture,net.usedWeightFlag,net.param);

mergedMixture
usedCompFlag
sum(net.usedWeightFlag)
