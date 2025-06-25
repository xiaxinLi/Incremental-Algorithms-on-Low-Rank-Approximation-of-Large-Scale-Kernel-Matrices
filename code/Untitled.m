% 与nystrom直接方法相比较
% 导入数据
Y=fea';
Y=Y/norm(Y,'fro');
[~,n]=size(Y);

%% 
% 初始化定义
desiredRank=50;mu=0;
param.type   = 'uni-sample';%选择取样方式
kernel.type  = 'Poly'; %选择核函数
kernel.par   = [2,0];

% 原样本X的svd分解求出U,D
n1=2000;n2=round(0.01*n1);%[自定义抽样个数]原样本个数为n1，每次增量的个数为n2
% n1=round(0.6*n);n2=round(0.01*n);  %[按样本比例抽取]
indx = randsample(1:n,n1);X = Y(:,indx);Y(:,indx)=[];%第一次取样为X
KernelMatrix =kernelmatrix(X,X,kernel); % X的核函数

% [U1,D1] = eigs(KernelMatrix, desiredRank);% 用svd求解U1,D1
% 用nystrom方法求解U1,D1
m=round(n1*0.1);% 抽取样本一般是原样本的0.1
[Z] = FindRep_us( X, m , param );%抽取样本 
[U,D]= NysDecom(X , Z , desiredRank , kernel);
L0=L;L1=L;L2=L;

E=norm(KernelMatrix - U *D* U','fro')./norm(KernelMatrix,'fro');

%% 增量开始，每次增量为n2
a=2;%表示增量迭代次数
time_nys=zeros(a,1);time_nys_incr=zeros(a,1);time_us=zeros(a,1);%t1用于统计nystrom方法的时间，t2用于统计us方法的时间
err_nys=zeros(a,1);err_nys_incr=zeros(a,1);err_us=zeros(a,1);%err用于统计nystrom方法的相对误差，err用于统计us方法的相对误差
for i=1:a
    indx = randsample(1:n-n1,n2);X_add = Y(:,indx);Y(:,indx)=[];%抽取的增量样本Y
    KernelMatrix_add =kernelmatrix([X,X_add],[X,X_add],kernel); % 增量后的核函数
    error = @(K) norm(KernelMatrix_add - K,'fro')./norm(KernelMatrix_add,'fro');%增量后的相对误差定义
    
    m=round((n1+n2)*0.1);% 抽取样本一般是原样本的0.1
    [Z] = FindRep_us( [X,X_add], m , param );%抽取样本 
    [U0,D0]= NysDecom([X,X_add] , Z , desiredRank , kernel);
    
%   nystrom直接
    t0=tic;
    [L0]=Incr_nys_2(L0,X,X_add,desiredRank,kernel,param);%nystrom增量方法 
    t0=toc(t0);
    time_nys(i,1)=t0;err_nys(i,1) = error(L0 * L0');
    
%   nystrom增量
    t1=tic;
    [U1,D1]=Incr_nys(U,D,X,X_add,desiredRank,kernel,param);%nystrom增量方法  
    t1=toc(t1);
    time_nys_incr(i,1)=t1;err_nys_incr(i,1) = error(U1*D1*U1');    

    t2=tic;
    [~,U2,D2]=Incr_nys_3(L,X,X_add,desiredRank,kernel,param);%nystrom增量方法  
    t2=toc(t2);
    time_us(i,1)=t2;err_us(i,1) = error(U2*D2*U2'); 
    
    n1=n1+n2;X=[X,X_add];
end
T=table(time_nys,time_nys_incr,time_us,err_nys,err_nys_incr,err_us);%表格数据用列排布