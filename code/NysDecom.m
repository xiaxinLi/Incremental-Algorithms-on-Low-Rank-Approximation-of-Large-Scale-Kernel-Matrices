 %The standard Nystrom method
function [U,D] = NysDecom(X , Z , desiredRank , kernel)
% [U , D] = NysDecom( X , Z , desiredRank , kernel ) computes eigenvalues
% and eigenvectors of the kernel matrix K.
% U is a matrix of size (n x desiredRank);D is a matrix of size (desiredRank x desiredRank) 
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

m = size(Z,2);
if size(X,1)~=size(Z,1), error('The given landmark set is not valid!'); end
if m < desiredRank, error('Select more landmark points!'); end

C =kernelmatrix(X,Z,kernel);
W =kernelmatrix(Z,Z,kernel);

W = (W + W')/2; % make sure W is symmetric
[Q , R] = qr (C, 0); % thin QR decomposition of matrix C
M = (R * pinv(full(W),1e-14) * R'); M = (M + M')/2;

[V,D] = eig(full(M));
[D,I] = sort(diag(D),'descend');
V = V(:, I);

U = Q * V(: , 1:desiredRank);
D = diag(D(1:desiredRank));
% L = U * sqrt(D);
end