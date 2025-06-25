% 最简单版本
function [U,D]=incr_update_pert(X,X_add,kernel,mu,U,D)
% 初始化定义
[~,m1]=size(U);[~,n2]=size(X_add);% 列为样本

% Step 1，计算K1'及K1'*U
K1=kernelmatrix(X_add,X,kernel); % 常用的核函数选择
K1=K1*U;

% Step 2，扰动框架求近似（此处为一阶扰动近似）
r1=D-mu*eye(m1);r1=ones(n2,m1)*r1;r1=K1./r1;
U=[U;r1]; %特征向量的计算
U=normc(real(U));
