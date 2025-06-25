function [U1,D1]=incr_update_pert_1(X,X_add,kernel,mu,U,D)
% 初始化定义
[n1,m1]=size(U);[~,n2]=size(X_add);% 列为样本

% Step 1，计算K1'及K1'*U
K1=kernel_Nystrom(X_add,X,kernel); % 常用的核函数选择
K1=[zeros(n1,m1);K1*U];
U=[U;zeros(n2,m1)];

% Step 2，扰动框架求近似（此处为一阶扰动近似）
U1=zeros(n1+n2,m1);D1=zeros(m1,m1);
for i=1:m1
    v=U(:,i);tmp=K1(:,i);
    %计算特征值近似
    D1(i,i)=D(i,i)+v'*tmp;
    
    %计算特征向量
    r2=D(i,i)*eye(m1)-D;r2(i,i)=1;r2=U/r2;r2(i,i)=0; %第二部分的计算
    tmp1=U'*tmp; r2=r2*tmp1; 
    r3=tmp-U*tmp1;r3=(1/(D(i,i)-mu))*r3; %第三部分的计算
    U1(:,i)=v+r2+r3;
end
U1=normc(real(U1));
