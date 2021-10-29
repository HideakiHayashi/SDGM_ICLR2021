clear
clc
dataSize = 20;
mu = [-0.75 -0.25 0.25 0.75 -0.75 -0.25 0.25 0.75 -0.75 -0.25 0.25 0.75 -0.75 -0.25 0.25 0.75; 0.75 0.75 0.75 0.75 0.25 0.25 0.25 0.25 -0.25 -0.25 -0.25 -0.25 -0.75 -0.75 -0.75 -0.75];
sigma = 0.125;
data = zeros(2,16*dataSize);
label = zeros(2,16*dataSize);
for i=1:16
    data(:,(i-1)*dataSize+1:i*dataSize) = mu(:,i)*ones(1,dataSize)+sigma*randn(2,dataSize);
    if i==1||i==3||i==6||i==8||i==9||i==11||i==14||i==16
        label(1,(i-1)*dataSize+1:i*dataSize) = ones(1,dataSize);
    else
        label(2,(i-1)*dataSize+1:i*dataSize) = ones(1,dataSize);
    end
end
data = data';
label = label';
save('sampleData.dat','-ascii','-tabs','data');
save('sampleLabels.dat','-ascii','-tabs','label');