%======================================
%======================================
function [] = plotScatterData(X,T,param)
Xt = X';

hold on;
marktype = ['+','*','s','x','d','^','v','>','<','p','h'];
markcolor = ['b','g','g','y','m','c','k','r','r','r','r','r','r','r','r'];
for c=1:param.classNum
    Xclass = Xt(logical(T(:,c)),:);
    scatter(Xclass(:,1),Xclass(:,2),marktype(c), markcolor(c))

end


hold off