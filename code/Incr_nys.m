%Algorithm 1 in the paper Incremental Algorithms on Low-Rank Approximation of Large-Scale
% Kernel Matrices Based on Perturbation of Invariant Subspaces

function [U,D]=Incr_nys(U,D,X,X_add,desiredRank,kernel,param)
%%Calculate W and C after the increments
n1=size(X,2);   n2=size(X_add,2);% The columns of the matrix are samples        
m=round(0.1*(n1+n2));  % The Nystrom method sampling ratio was 0.1

[~,supp] = FindRep_us([X,X_add] , m , param );% Select sampling method
    
supp1=supp(supp<=n1); supp2=supp(supp>n1);
%Classification, supp1 is sampled in X, supp2 is sampled in Y 

inc_Y=[X,X_add];
%Calculate the first part of the post-incremental kernel matrix, Approximation of k(X1,X1)
K1=U';K1=K1(:,supp1);K1=diag(D).*K1; K1=U*K1;
%Calculate the third part of the post-incremental kernel matrix, k(Y,X1)
X1=X(:,supp1);K3=kernelmatrix(X_add,X1,kernel);
%Calculate the second part of the post-incremental kernel matrix, k(X,Y1)
Y1=inc_Y(:,supp2);
K2=kernelmatrix(X,Y1,kernel);
%Calculate the fourth part of the post-incremental kernel matrix, k(Y,Y1)
K4=kernelmatrix(X_add,Y1,kernel);

C=[K1,K2;K3,K4];
W=C(supp,:);
W= (W+W')/2;  % casts W into a symmetric semi-positive definite matrix

%% Use the calculated C and W to calculate the approximated of U¡¢D
% Eigenvalue Decomposition
[UW,SW] = eig(full(W));[SW,I] = sort(diag(SW),'descend');
UW = UW(:, I);
D=diag(SW(1:desiredRank,:)*(n1+n2)/m);

SW = 1 ./ SW(1:desiredRank,:);
UW = UW(:,1:desiredRank).* SW';
U =sqrt(m/(n1+n2)) * C* UW; 
end
