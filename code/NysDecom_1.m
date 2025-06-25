 %列为样本数量，用pinv求W的广义逆且tol=1e-10
function [U,D] = NysDecom_1(X , Z , desiredRank , kernel)
% [U , D] = NysDecom( X , Z , desiredRank , kernel ) computes eigenvalues
% and eigenvectors of the kernel matrix K.
% L is a matrix of size (nxdesiredRank) which gives an approximation of the kernel matrix K,
% in the form of L * L'. 
%
% Note: Use 'FindRep.m' to find the landmark set Z.  
%
% Farhad Pourkamali-Anaraki, E-mail: Farhad.Pourkamali@colorado.edu
% University of Colorado Boulder
%{
Inputs:
    - X: input data matrix of size pxn, where p is the dimension and n is
    the number of samples
    - Z: landmark matrix of size pxm, where m is the number of landmark
    points (Use FindRep.m to find the landmark points)
    - desiredRank: target rank
    - kernel.type: kernel type
        1) 'RBF': Gaussian - k(x,y) = exp(-gamma.*|x-y|_2^2) ,gamma= 1/(2*sigma^2); [parameter: sigma]
        2) 'Poly': Polynomial - k(x,y) = (x'y+c).^d [parameters: c & d]
    - kernel.par: parameters for kernels, sigma, c and d. For polynomial
    kernels, the order should be [degree d,constant c]. 
%}

m = size(Z,2);n = size(X,2);
if size(X,1)~=size(Z,1), error('The given landmark set is not valid!'); end
if m < desiredRank, error('Select more landmark points!'); end

C =kernelmatrix(X,Z,kernel);
W =kernelmatrix(Z,Z,kernel);

%% 利用计算出的C和W求新的L
% Eigenvalue Decomposition
[UW,SW] = eig(full(W));[SW,I] = sort(diag(SW),'descend');
UW = UW(:, I);
D=diag(SW(1:desiredRank,:)*n/m);

% t=min(length(SW>1e-14),desiredRank);%desireRank与SW中非0(小于1e-12看作0)的特征值
SW = 1 ./ SW(1:desiredRank,:); 
UW = bsxfun(@times , UW(:,1:desiredRank), SW');
% C = bsxfun(fun,A,B) 对数组 A 和 B 应用函数句柄 fun 指定的按元素二元运算
% fun=times表示数组乘法.*
U =sqrt(m/n) * C* UW; % approximated by L * L'
end


