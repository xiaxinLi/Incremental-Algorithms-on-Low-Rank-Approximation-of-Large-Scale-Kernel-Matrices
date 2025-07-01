%Algorithm 2 in the paper Incremental Algorithms on Low-Rank Approximation of Large-Scale
% Kernel Matrices Based on Perturbation of Invariant Subspaces
function [U,D]=incr_update_pert(X,X_add,kernel,mu,U,D)
% Initialize the definition
[~,m1]=size(U);[~,n2]=size(X_add);

% Step 1£¬Calculate K1' and K1'*U
K1=kernelmatrix(X_add,X,kernel); % Calculate K1
K1=K1*U;

% Step 2£¬Calculate the approximation of the kernel matrix
r1=D-mu*eye(m1);r1=ones(n2,m1)*r1;r1=K1./r1;
U=[U;r1]; 
U=normc(real(U));
