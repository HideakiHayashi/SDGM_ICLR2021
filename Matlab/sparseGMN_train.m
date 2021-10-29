%======================================
%Training of Sparse GMN
%======================================

function [net] = sparseGMN_train(inTrainDataName,inTrainLabelName,inClassNum,inMaxCompNum,iskernel)


if nargin ~= 5
    disp('Not enough arguments!');
    net = [];
    return
end

%=========================================

%=========================================
classNum = inClassNum;                  
maxCompNum = inMaxCompNum;              
tCompNum = classNum*maxCompNum;         
min_converge = 1e-5;                    
amax = 1e5;                             
minDiffr = 1e-3;                        
TrainDataName = inTrainDataName;        
TrainLabelName = inTrainLabelName;      
%=========================================

%=============
%=============
X = load(TrainDataName);        
X = X';
T = load(TrainLabelName);       

%=============

%=============
if iskernel
    N = size(X,2);                          
    H = N+1;                                
else
    [D N] = size(X);
    H = 1+D*(D+3)/2;                        
end

pastCompNum = zeros(classNum,1);
for c=2:classNum
    pastCompNum(c) = pastCompNum(c-1)+maxCompNum;
end
usedWeightNum = (tCompNum-1)*H;             

param = struct('classNum',classNum,'maxCompNum',maxCompNum,'tCompNum',tCompNum,'N',N,'H',H,'pastCompNum',pastCompNum,'usedWeightNum',usedWeightNum);
trainDataNumClass = sum(T);

repT = zeros(param.N,param.tCompNum);
for c=1:classNum
    repT(:,pastCompNum(c)+1:pastCompNum(c)+maxCompNum) = repmat(T(:,c),1,maxCompNum);
end

% plotScatterData(X,T,param);
% drawnow;

%=============

%=============
w = zeros((param.tCompNum-1)*H,1);                      
a = ones((param.tCompNum-1)*H,1);                       
mixture = ones(param.classNum,param.maxCompNum)./param.maxCompNum; 
usedWeightFlag = true(param.usedWeightNum,1);           
r = initializePostMix(X, T, param);                     

%=============

%=============
if iskernel
    Phi = calcGramMatrix(X,X); 
else
    Phi = nonlinearTrans(X');  
end

%=============

%=============
w_used = w;   
a_used = a;   
diffsum = amax; 
count = 1;
while diffsum > min_converge
    
    A = diag(a_used);  
    [O2, Y] = forward(w, Phi, mixture, param, usedWeightFlag);	
    diffrsum = amax;                                            
    while diffrsum > minDiffr
        
        disp('M step');
        gradSum = amax;    
        gradSumOld = amax;
        w_ini = w_used;
        learnRate = 1.0;
        while gradSum > min_converge
            grad = calcGrad(T,r,O2,Phi,usedWeightFlag,param) - A*w_used;   
            Hess = calc_Hess_ww(O2,Phi,usedWeightFlag,param) + A;          
            
            gradSum = sqrt(grad'*grad);
            disp(gradSum);
            
            if gradSum>gradSumOld
                w_used = w_ini;
                learnRate = learnRate*0.5;
                gradSumOld = amax;
            else
                delta_w = Hess\grad;       
                w_used = w_used + learnRate*delta_w;  
                gradSumOld = gradSum;
            end
            w(usedWeightFlag) = w_used;
            [O2, Y] = forward(w, Phi, mixture, param, usedWeightFlag);
        end
        
        r_new = calcPostMix(O2, Y, param);
        r_new = r_new.*repT;                
        diffr = r_new-r;
        diffrsum = sum(sum(abs(diffr)))./(tCompNum*N);
        disp('E step');
        disp(diffrsum);
        r = r_new;
    end
    
    
    
    Sigma = inv(Hess);         
    disp('Update hyperparameters');
    gamma = 1.0 - a_used.*diag(Sigma);
    newa = gamma./(w_used.^2);
    newa(newa >= amax) = amax;
    diff = newa - a_used;
    diffsum = diff'*diff;
    disp(diffsum);
    a_used = newa;
    
    a(usedWeightFlag) = a_used;
    w(usedWeightFlag) = w_used;
    
    
    for c=1:param.classNum
        for m=1:param.maxCompNum
            mixture(c,m) = sum(r(logical(T(:,c)),param.pastCompNum(c)+m))./trainDataNumClass(c);
        end
    end
    
    
    
    usedCompFlag = mixture>min_converge;
    usedCompFlagReshape = reshape(usedCompFlag',[1,param.tCompNum]);
    usedCompFlagReshape(end) = [];
    usedCompFlagRep = repmat(usedCompFlagReshape,param.H,1);
    usedCompFlagRep = reshape(usedCompFlagRep,[(param.tCompNum-1)*param.H, 1]);
    usedA = a<amax;
    usedWeightFlag = logical(usedA.*usedCompFlagRep);
    param.usedWeightNum = sum(usedWeightFlag);
    if param.usedWeightNum == 0
        break;
    end
    w_used = w(usedWeightFlag);
    a_used = a(usedWeightFlag);
    
    
%     if count < 10 || rem(count, 10)==0
%         clf;
%         plotDecisionBoundaryComp(w,mixture,X,T,usedWeightFlag,param);
%         drawnow;
%         figname = sprintf('Boundary%d.fig', count);
%         savefig(figname);
%     end
%     count = count + 1;
end

%=============

%=============
TrainingError = calcClassificationError(Y,T);
display(TrainingError);

%=============

%=============
data = struct('X',X,'T',T);
configure = struct('min_converge',min_converge,'amax',amax,'minDiffr',minDiffr,'TrainDataName',TrainDataName,'TrainLabelName',TrainLabelName);
net = struct('w',w,'a',a,'mixture',mixture,'usedWeightFlag',usedWeightFlag,'Y',Y,'O2',O2,'r',r,'param',param,'configure',configure,'TrainingError',TrainingError,'iskernel',iskernel,'data',data);
