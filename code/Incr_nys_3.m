% 利用上一次的L的信息，更新迭代新的核矩阵K---重采样
function [L,U,D]=Incr_nys_3(L,X,X_add,desiredRank,kernel,param)
%%求增量后的W和C
n1=size(X,2);   n2=size(X_add,2);% 列为样本        
m=round(0.1*(n1+n2));  % nystrom方法抽样比例为0.1
% supp=randperm(n1+n2,m);  %随机抽样
[~,supp] = FindRep_us([X,X_add] , m , param );% 其它的抽样方式
    
supp1=supp(supp<=n1); supp2=supp(supp>n1);%分类，supp1在X里抽样，supp2在Y里抽样 

inc_Y=[X,X_add];
%求第一块
K1=L';K1=K1(:,supp1);K1=L*K1;
%求第三块
X1=X(:,supp1);K3=kernelmatrix(X_add,X1,kernel);%核矩阵k(Y,X1)
%求第二块
Y1=inc_Y(:,supp2);
K2=kernelmatrix(X,Y1,kernel);%核矩阵k(X,Y1)
%求第四块
K4=kernelmatrix(X_add,Y1,kernel);%核矩阵k(Y,Y1)

C=[K1,K2;K3,K4];
W=C(supp,:);
W= (W+W')/2;  %将W强制转换成对称半正定矩阵

%% 利用计算出的C和W求新的L
% Eigenvalue Decomposition
[UW,SW] = eig(full(W));[SW,I] = sort(diag(SW),'descend');
UW = UW(:, I);
D=diag(SW(1:desiredRank,:));

SW = 1 ./ SW(1:desiredRank,:);
UW = bsxfun(@times , UW(:,1:desiredRank), SW');
% C = bsxfun(fun,A,B) 对数组 A 和 B 应用函数句柄 fun 指定的按元素二元运算
% fun=times表示数组乘法.*
U  = C* UW; % approximated by L * L'
L  = U*D*U';
end
