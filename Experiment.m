%Comparative experiments
% For specific experimental descriptions and results, please refer to the paper Incremental Algorithms on Low-Rank Approximation of Large-Scale
% Kernel Matrices Based on Perturbation of Invariant Subspaces
%% Set up workspace 
clear;clc;
%Add code to the path
addpath('D:\Code\students\Master\2022\李夏昕\increment\code')
addpath('D:\Code\students\Master\2022\李夏昕\increment\code\Contrastive')
addpath('D:\Code\students\Master\2022\李夏昕\increment\experiment')

%% 
% Initialize the definition
mu=0;
param.type   = 'uni-sample';%the choice of sampling method for the FindRep_us function

for z=2
    if z==1
        kernel.type  = 'Poly'; %Select the kernel function[kernelmatrix funcation]
        kernel.par   = [2,0];
        kFunc = @(X,rowInd,colInd) polynomialKernel(X,rowInd,colInd,kernel);
%        folderPath = 'G:\matlab\all\all\xiu\DXS';
        folderPath = 'D:\Code\students\Master\2022\李夏昕\xiu\DXS';
    else
        kernel.type  = 'RBF';%Select the kernel function[kernelmatrix funcation]
        kernel.par   = 0.1;
        gamma=1/(2*kernel.par^2);
        kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma);
%        folderPath = 'G:\matlab\all\all\xiu\GS';
        folderPath = 'D:\Code\students\Master\2022\李夏昕\xiu\GS';
    end
a=10;%Represents the total number of incremental iterations
numbers=5;%Indicates the number of times the experiment was repeated
%% Start the experiment
for l=1:4
    tables = cell(1, numbers);tables_mean=zeros(a,12);
    for j=1:numbers 
    %Import data
    if l==1
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\AR_64x64.mat');
        n1=2000;% The number of X'samples is n1
        desiredRank=20;
        fileName = 'AR.mat';% Specify the name of the file you want to save
    elseif l==2
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\CIFAR-100.mat');
        n1=20000;
        desiredRank=50;
        fileName = 'CIFAR-100.mat';
    elseif l==3
        data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\MNIST.mat');
        n1=50000;
        desiredRank=100;
        fileName = 'MNIST.mat';
    elseif l==4
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\FounderType-17_64x64.mat');
        n1=80000;
        desiredRank=100;
        numbers=3;
        fileName = 'FounderType.mat';
    else
         data = importdata('D:\Code\students\Master\2022\李夏昕\increment\data\YouTubeFace_32x32.mat');
        n1=150000;
        desiredRank=50;
        fileName = 'YouTubeFace.mat';
    end
    % Build the full file path to be saved    
    fullFilePath = fullfile(folderPath, fileName);
                
        Y=data.fea';Y=double(Y);
        Y=Y/norm(Y,'fro');
        [~,n]=size(Y);
        n_add=round(0.01*n1);%[Custom sampling ratio, 0.01 in this case] The number of increments is n2
        indx = randsample(1:n,n1);X = Y(:,indx);Y(:,indx)=[];%The original sample is X
        
% Calculate the eigenvalues and eigenvectors of the kernel matrix composed of sample X 
% for the initial input of algorithm 1 and algorithm 2, i.e., the Nnstrom method
        m=round(n1*0.1);
        [Z] = FindRep_us( X, m , param ); 
        [U,D]= NysDecom(X , Z , desiredRank , kernel);
        U6=U;U5=U;D6=D;D5=D;
        
%% Increments start with each increment being n2
        time_nys=zeros(a,1);time_nys_kmeanspp=zeros(a,1);time_nys_RAS=zeros(a,1);
        time_nys_REC=zeros(a,1);time_alg1=zeros(a,1);time_alg2=zeros(a,1);
        %Used to record the time of each method
        err_nys=zeros(a,1);err_nys_kmeanspp=zeros(a,1);err_nys_RAS=zeros(a,1);
        err_nys_REC=zeros(a,1);err_alg1=zeros(a,1);err_alg2=zeros(a,1);
        %Used to record the relative error of each method
        for i=1:a
            n2=n_add*i;
            indx = randsample(1:n-n1,n2);X_add = Y(:,indx);%Increment sample Y
            KernelMatrix_add =kernelmatrix([X,X_add],[X,X_add],kernel); % Kernel matrix after increment
            error = @(K) norm(KernelMatrix_add - K,'fro')./norm(KernelMatrix_add,'fro');%Define the relative error

            % The standard Nystrom method
            t1=tic;
            [Z] = FindRep_us( [X,X_add], round((n1+n2)*0.1) , param );%抽取样本
            [U1,D1] = NysDecom([X,X_add] , Z , desiredRank, kernel); % standard Nystrom method
            t1=toc(t1);
            time_nys(i,1)=t1;err_nys(i,1) = error(U1 * D1 * U1');
         
            %nystrom-kmeans++ sampling method
            if l==1
              t2=tic;
              max_iters=10;
              [Z,~] = kmeanspp([X,X_add], round((n1+n2)*0.1),max_iters);
            %Nystrom method
              [U2,D2] = NysDecom([X,X_add] , Z , desiredRank, kernel); % standard Nystrom method
              t2=toc(t2);
              time_nys_kmeanspp(i,1)=t2;err_nys_kmeanspp(i,1) = error(U2 * D2 * U2');
            else
                time_nys_kmeanspp(i,1)=0;err_nys_kmeanspp(i,1) = 0;
            end
        
            %the random adaptive sampling method
            t3=tic;
            bw= 0.1;% Gamma = 1/(bw^2) in stochastic Fourier transform; 
            RAS_epsilon = 1e-10;
            RAS_c = 200*RAS_epsilon;
            RAS_lambda = 10^(-6);
            RAS_t = 0;RAS_nb_FF = round((n1+n2)*0.1);RAS_updating = true;
            RAS_idS = RAS_RFF([X,X_add],bw,RAS_c,RAS_t,RAS_epsilon,RAS_lambda,RAS_nb_FF,RAS_updating);%抽取样本
            [Z]=[X,X_add];Z=Z(:,RAS_idS);
             %Nystrom method
            [U3,D3] = NysDecom([X,X_add] , Z , desiredRank, kernel); % standard Nystrom method
            t3=toc(t3);
            time_nys_RAS(i,1)=t3;err_nys_RAS(i,1) = error(U3 * D3 * U3');
            
            %Recursive Sampling for the Nystrom Method
            t4=tic;
            s=round((n1+n2)*0.1);
            [C4,W4]=recursiveNystrom([X,X_add]',s,kFunc);
            t4=toc(t4);
            time_nys_REC(i,1)=t4;err_nys_REC(i,1) = error(C4 * W4 * C4');
            
            %Algorithm 1 in the paper
            t5=tic;
            [U5,D5]=Incr_nys(U5,D5,X,X_add,desiredRank,kernel,param);%nystrom增量方法  
            t5=toc(t5);
            time_alg1(i,1)=t5;err_alg1(i,1) = error(U5 * D5 * U5');
            
            %Algorithm 2 in the paper
            t6=tic;
            [U6,D6]=incr_update_pert(X,X_add,kernel,mu,U6,D6);
            t6=toc(t6);
           time_alg2(i,1)=t6;err_alg2(i,1)=error(U6 * D6 * U6');
            
            n1=n1+n2;X=[X,X_add];
        end
         tables{1,j}=[time_nys,time_nys_kmeanspp,time_nys_RAS,time_nys_REC,time_alg1,time_alg2,err_nys,err_nys_kmeanspp,err_nys_RAS,err_nys_REC,err_alg1,err_alg2];
         tables_mean=tables{j}+tables_mean;    
    end
    tables_mean=tables_mean/numbers;
    save(fullFilePath, 'tables','tables_mean');
    % Use the save function to save the cell array to the specified file   
end

end