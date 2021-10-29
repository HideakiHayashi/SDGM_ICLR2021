%======================================
%Initialize the posterior probability of mixture using k-means algorithm
%======================================
function [r] = initializePostMix(X, T, param)
X = X';
dataLabel = logical(T);
I = eye(param.maxCompNum);
r = zeros(param.N, param.tCompNum);

for c=1:param.classNum
    dataLabelClass = dataLabel(:,c);
    dataClass = X(dataLabelClass,:);
    [centroid, pointsInCluster, assignment] = myKmeans(dataClass, param.maxCompNum);
    oneOfKassignment = I(assignment,:);
    r(dataLabelClass,param.pastCompNum(c)+1:param.pastCompNum(c)+param.maxCompNum) = oneOfKassignment;
end