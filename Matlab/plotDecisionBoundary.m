%======================================
%======================================
function [] = plotDecisionBoundary(w,mixture,X,T,usedWeightFlag,param)
Xt = X';


hold on;
marktype = ['+','*','s','x','d','^','v','>','<','p','h'];
markcolor = ['b','g','g','y','m','c','k','r','r','r','r','r','r','r','r'];
for c=1:param.classNum
    Xclass = Xt(logical(T(:,c)),:);
    scatter(Xclass(:,1),Xclass(:,2),marktype(c),markcolor(c))
    for m=1:param.maxCompNum
        if c~=param.classNum||m~=param.maxCompNum
            tmpUsed = usedWeightFlag(param.pastCompNum(c)*param.H+(m-1)*param.H+1:param.pastCompNum(c)*param.H+m*param.H);
            tmpUsed(end) = [];
            Relevance = Xt(tmpUsed,:);
            scatter(Relevance(:,1),Relevance(:,2),'ro');
        end
    end
end

box = 1.1*[min(Xt(:,1)) max(Xt(:,1)) min(Xt(:,2)) max(Xt(:,2))];
axis(box);
gsteps		= 50;
range1		= box(1):(box(2)-box(1))/(gsteps-1):box(2);
range2		= box(3):(box(4)-box(3))/(gsteps-1):box(4);
[grid1 grid2]	= meshgrid(range1,range2);
Xgrid		= [grid1(:) grid2(:)];
param.N = size(Xgrid,1);

Xgrid = Xgrid';
Phi_grid = zeros(param.N,param.H);
for h=1:param.H-1
    for n=1:param.N
        Phi_grid(n,h) = kernel(Xgrid(:,n),X(:,h));
    end
end
Phi_grid(:,param.H) = ones(param.N,1);

[O2_grid, Y_grid] = forward(w, Phi_grid, mixture, param, usedWeightFlag);

for c=1:param.classNum-1
    contour(range1,range2,reshape(Y_grid(:,1),size(grid1)),[0.5 0.5],'k-','LineWidth',2);
end
hold off