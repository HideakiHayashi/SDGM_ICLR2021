%======================================
%Leave one out cross validation using Sparse GMN
%======================================
close all;
clear;
clc;

DataName = 'testData.dat';       
LabelName = 'testLabels.dat';    
inClassNum = 2;                  
inMaxCompNum = 1;                
iskernel = true;

data = load(DataName);
labels = load(LabelName);
[N D] = size(data);
accuracy = zeros(N,1);
mkdir('LOOCVdata');
for n=1:N
    %test data
    testData = data(n,:);
    testDataName = sprintf('./LOOCVdata/testData%d.dat',n);
    save(testDataName,'-ascii','-tabs','testData');
    %test label
    testLabel = labels(n,:);
    testLabelName = sprintf('./LOOCVdata/testLabel%d.dat',n);
    save(testLabelName,'-ascii','-tabs','testLabel');
    %training data
    trainData = data;
    trainData(n,:) = [];
    trainDataName = sprintf('./LOOCVdata/trainData%d.dat',n);
    save(trainDataName,'-ascii','-tabs','trainData');
    %training label
    trainLabel = labels;
    trainLabel(n,:) = [];
    trainLabelName = sprintf('./LOOCVdata/trainLabel%d.dat',n);
    save(trainLabelName,'-ascii','-tabs','trainLabel');

    net = sparseGMN_train(trainDataName,trainLabelName,inClassNum,inMaxCompNum,iskernel);
    trainResults(n) = net;
    
    result = sparseGMN_classify(net,testDataName,testLabelName);
    testResults(n) = result;
    
    accuracy(n) = result.TestingError;
    meanError = mean(accuracy);
end
save('./LOOCVdata/trainResults.mat','trainResults');
save('./LOOCVdata/testResults.mat','testResults');

%plotDecisionBoundary(net.w,net.mixture,net.data.X,net.data.T,net.usedWeightFlag,net.param);
[mergedMixture,usedCompFlag] = mergeMixture(net.mixture,net.usedWeightFlag,net.param);

