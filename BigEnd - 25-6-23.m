%增量实验
%% Set up workspace 
clear;clc;

addpath('D:\Code\students\Master\2022\李夏昕\increment\code')
addpath('D:\Code\students\Master\2022\李夏昕\increment\code\Contrastive')
addpath('D:\Code\students\Master\2022\李夏昕\increment\experiment')
%addpath('G:\matlab\all\all\increment\code')
%addpath('G:\matlab\all\all\increment\code\Contrastive')
%addpath('G:\matlab\all\all\increment\experiment')
%addpath('G:\matlab\all\all\increment\data')
%% 
% 初始化定义
% 66行需要改 n1
mu=0;
param.type   = 'uni-sample';%选择取样方式

for z=2
    if z==1
        kernel.type  = 'Poly'; %选择核函数
        kernel.par   = [2,0];
        kFunc = @(X,rowInd,colInd) polynomialKernel(X,rowInd,colInd,kernel);
%        folderPath = 'G:\matlab\all\all\xiu\DXS';
        folderPath = 'D:\Code\students\Master\2022\李夏昕\xiu\DXS';
    else
        kernel.type  = 'RBF'; %选择核函数
        kernel.par   = 0.1;
        gamma=1/(2*kernel.par^2);
        kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma);
%        folderPath = 'G:\matlab\all\all\xiu\GS';
        folderPath = 'D:\Code\students\Master\2022\李夏昕\xiu\GS';
    end
%     dist = pdist2(X, X); % 计算训练向量之间的欧氏距离矩阵
%     kernel.par = sum(dist(:)) / (n1 * (n1-1)); % 计算平均距离
a=5;%表示增量迭代的总次数
numbers=5;%表示重复实验的次数
%% 实验开始
for l=1:4
    tables = cell(1, numbers);tables_mean=zeros(a,12);
    for j=1:numbers 
    %导入数据
    if l==1
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\AR_64x64.mat');
%        data = importdata('G:\matlab\all\all\increment\data\AR_64x64.mat');
        n1=2000;
        desiredRank=20;
        fileName = 'AR.mat';% 指定要保存的文件名
    elseif l==2
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\CIFAR-100.mat');
%        data = importdata('G:\matlab\all\all\increment\data\CIFAR-100.mat');
        n1=20000;
        desiredRank=50;
        fileName = 'CIFAR-100.mat';% 指定要保存的文件名
    elseif l==3
        data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\MNIST.mat');
%         data = importdata('G:\matlab\all\all\increment\data\MNIST.mat');
        n1=50000;
        desiredRank=100;
        fileName = 'MNIST.mat';% 指定要保存的文件名
    elseif l==4
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\FounderType-17_64x64.mat');
%        data = importdata('G:\matlab\all\all\increment\data\FounderType-17_64x64.mat');
        n1=80000;
        desiredRank=100;
        numbers=3;
        fileName = 'FounderType.mat';% 指定要保存的文件名
    else
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\YouTubeFace_32x32.mat');
%        data = importdata('G:\matlab\all\all\increment\data\YouTubeFace_32x32.mat');
        n1=150000;
        numbers=3;%表示重复实验的次数
        desiredRank=50;
        fileName = 'YouTubeFace.mat';% 指定要保存的文件名
    end
    % 构建要保存的完整文件路径    
    fullFilePath = fullfile(folderPath, fileName);
    
             
        Y=data.fea';Y=double(Y);
        Y=Y/norm(Y,'fro');
        [~,n]=size(Y);
%         n1=2000;
        n_add=round(0.02*n1);%[自定义抽样个数]原样本个数为n1，每次增量的个数为n2
% n1=round(0.6*n);n2=round(0.01*n);  %[按样本比例抽取]
        indx = randsample(1:n,n1);X = Y(:,indx);Y(:,indx)=[];%第一次取样为X
%        KernelMatrix =kernelmatrix(X,X,kernel); % X的核函数
% [U1,D1] = eigs(KernelMatrix, desiredRank);% 用svd求解U1,D1
% 用nystrom方法求解U1,D1 

        m=round(n1*0.1);% 抽取样本一般是原样本的0.1
        [Z] = FindRep_us( X, m , param );%抽取样本 
        [U,D]= NysDecom(X , Z , desiredRank , kernel);
        U6=U;U5=U;D6=D;D5=D;
%% 增量开始，每次增量为n2
        time_nys=zeros(a,1);time_nys_kmeanspp=zeros(a,1);time_nys_RAS=zeros(a,1);
        time_nys_incr=zeros(a,1);time_us=zeros(a,1);time_nys_REC=zeros(a,1);
        %t1用于统计nystrom方法的时间，t2用于统计us方法的时间
        err_nys=zeros(a,1);err_nys_kmeanspp=zeros(a,1);err_nys_RAS=zeros(a,1);
        err_nys_incr=zeros(a,1);err_us=zeros(a,1);err_nys_REC=zeros(a,1);
        %err用于统计nystrom方法的相对误差，err用于统计us方法的相对误差
        for i=1:a
            n2=n_add*i;
            indx = randsample(1:n-n1,n2);X_add = Y(:,indx);%抽取的增量样本Y
            KernelMatrix_add =kernelmatrix([X,X_add],[X,X_add],kernel); % 增量后的核函数
            error = @(K) norm(KernelMatrix_add - K,'fro')./norm(KernelMatrix_add,'fro');%增量后的相对误差定义
                  
            %   us
            t6=tic;
            [U6,D6]=incr_update_pert(X,X_add,kernel,mu,U6,D6);
            t6=toc(t6);
           time_us(i,1)=t6;err_us(i,1)=error(U6 * D6 * U6');
%            save('D:\Code\students\Master\2022\李夏昕\xiu-reslut\1\example.mat','time_us','err_us','-append'); 
            %   nystrom增量
            t5=tic;
            [U5,D5]=Incr_nys(U5,D5,X,X_add,desiredRank,kernel,param);%nystrom增量方法  
            t5=toc(t5);
            time_nys_incr(i,1)=t5;err_nys_incr(i,1) = error(U5 * D5 * U5');
%             save('D:\Code\students\Master\2022\李夏昕\xiu-reslut\1\example.mat','time_nys_incr','err_nys_incr','-append');
            %   nystrom直接
            t1=tic;
            [Z] = FindRep_us( [X,X_add], round((n1+n2)*0.1) , param );%抽取样本
            [U1,D1] = NysDecom([X,X_add] , Z , desiredRank, kernel); % standard Nystrom method
            t1=toc(t1);
            time_nys(i,1)=t1;err_nys(i,1) = error(U1 * D1 * U1');
%             save('D:\Code\students\Master\2022\李夏昕\xiu-reslut\1\example.mat','time_nys','err_nys','-append');
            
            %nystrom-kmeans++采样
%               t2=tic;
            %选择取样方式--RAS采样
%              max_iters=10;
%              [Z,~] = kmeanspp([X,X_add], round((n1+n2)*0.1),max_iters);
            %Nystrom方法
%              [U2,D2] = NysDecom([X,X_add] , Z , desiredRank, kernel); % standard Nystrom method
%              t2=toc(t2);
%              time_nys_kmeanspp(i,1)=t2;err_nys_kmeanspp(i,1) = error(U2 * D2 * U2');
        
            %nystrom-RAS采样
            t3=tic;
            %选择取样方式--RAS采样
            bw= 0.1;% 随机傅里叶变换中gamma = 1/(bw^2); 
            RAS_epsilon = 1e-10;
            RAS_c = 200*RAS_epsilon;
            RAS_lambda = 10^(-6);
            RAS_t = 0;RAS_nb_FF = round((n1+n2)*0.1);RAS_updating = true;
        
            RAS_idS = RAS_RFF([X,X_add],bw,RAS_c,RAS_t,RAS_epsilon,RAS_lambda,RAS_nb_FF,RAS_updating);%抽取样本
            [Z]=[X,X_add];Z=Z(:,RAS_idS);
             %Nystrom方法
            [U3,D3] = NysDecom([X,X_add] , Z , desiredRank, kernel); % standard Nystrom method
            t3=toc(t3);
            time_nys_RAS(i,1)=t3;err_nys_RAS(i,1) = error(U3 * D3 * U3');
            
            %Recursive Sampling for the Nystr?m Method
            t4=tic;
            s=round((n1+n2)*0.1);
            [C4,W4]=recursiveNystrom([X,X_add]',desiredRank,s,kFunc);
            t4=toc(t4);
            time_nys_REC(i,1)=t4;err_nys_REC(i,1) = error(C4 * W4 * C4');
            
            n1=n1+n2;X=[X,X_add];
        end
         tables{1,j}=[time_nys,time_nys_kmeanspp,time_nys_RAS,time_nys_REC,time_nys_incr,time_us,err_nys,err_nys_kmeanspp,err_nys_RAS,err_nys_REC,err_nys_incr,err_us];%数据用列排布
         tables_mean=tables{j}+tables_mean;    
    end
    save(fullFilePath, 'tables','tables_mean');
    % 使用 save 函数保存元胞数组到指定文件
%     tables_mean=tables_mean/numbers;
    
end

end