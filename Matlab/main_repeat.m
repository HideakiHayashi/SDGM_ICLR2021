%======================================
%Repeating hold out validation for Banana data using SDGM
%======================================
close all;
clear;
clc;

% ==============
% Parameters
% ==============
inDirName = 'BananaData';
outDirName = 'BananaResults';
startNum = 1;
repNum = 100;
inClassNum = 2;                     %Number of classes
inMaxCompNum = 3;                   %Max number of components
iskernel = true;
% ==============
mkdir(outDirName);
mkdir(strcat(outDirName, '/mat'))

errors = zeros(repNum,1);
nWeights = zeros(repNum,1);
nComps = zeros(repNum,1);

for n=startNum:repNum
    fprintf('Trial%d',n);
    % Training
    trainDataName = sprintf('%s/trainData%d.dat',inDirName,n);
    trainLabelName = sprintf('%s/trainLabels%d.dat',inDirName,n);
    net = sparseGMN_train(trainDataName,trainLabelName,inClassNum,inMaxCompNum,iskernel);
    save(sprintf('%s/mat/trainResults%d.mat',outDirName,n),'net');
    
    
    % Test
    testDataName = sprintf('%s/testData%d.dat',inDirName,n);
    testLabelName = sprintf('%s/testLabels%d.dat',inDirName,n);
    result = sparseGMN_classify(net,testDataName,testLabelName);
	save(sprintf('%s/mat/testResults%d.mat',outDirName,n),'result');
    
    % Summary
    errors(n) = result.TestingError;
    nWeights(n) = sum(net.usedWeightFlag);
    [mergedMixture,usedCompFlag] = mergeMixture(net.mixture,net.usedWeightFlag,net.param);
    nComps(n) = sum(sum(usedCompFlag));
    clear net;
    clear result;
    
end

save(strcat(outDirName,'/TestError.dat'),'errors','-ascii','-tabs');
save(strcat(outDirName,'/nWeights.dat'),'nWeights','-ascii','-tabs');
save(strcat(outDirName,'/nComps.dat'),'nComps','-ascii','-tabs');



