function C =kernelmatrix(X,Z,kernel)
%{
Inputs:
    - X: input data matrix of size txN, where t is the dimension and N is
    the number of samples
    - Z: landmark matrix of size txm, where m is the number of landmark
    - desiredRank: target rank
    - kernel_type: kernel type
        1) 'RBF': Gaussian - k(x,y) = exp(-gamma.*|x-y|_2^2),gamma= 1/(2*sigma^2); [parameter: sigma]
        2) 'Poly': Polynomial - k(x,y) = (x'y+c).^d [parameters: c & d]
    - kernel.par: parameters for kernels, gamma, c and d. For polynomial
    kernels, the order should be [degree d,constant c]. 
%}

if size(X,1)~=size(Z,1), error('The given landmark set is not valid!'); end

% start switch
switch kernel.type
    case 'RBF'
        gamma = 1/(2*kernel.par^2);
        C = exp( -gamma.*sqdist(X,Z) ); % C: n*m 
    case 'Poly'
        d = kernel.par(1); c = kernel.par(2); 
        C = (X' * Z + c).^d; % C: n*m 
    case 'laplace'
        X=X';Z=Z';
        bandwidth=kernel.par;
        C = exp(-pdist2(X,Z,"minkowski",1)/bandwidth);
end % end switch 
end

function [ Dist ] = sqdist( A , B )
% sqdist: Squared Euclidean distances between columns of A and B.
% Thus, data points are assumed to be in columns, not rows
%
% Dist = sqdist( A, B )
% Dist = sqdist( A ) assumes B = A

% Farhad Pourkamali-Anaraki, E-mail: Farhad.Pourkamali@colorado.edu
% University of Colorado Boulder

%{
Inputs: 
    - A: input matrix of size p*n1
    - B: input matrix of size p*n2 
Outputs: 
    - Dist: matrix of pairwise distances of size n1*n2
%}

if nargin < 2
    % Assume B = A
    AA = sum(A.^2,1); 

    Dist = -2*(A'*A);
    Dist = bsxfun( @plus, Dist, AA' );
    Dist = bsxfun( @plus, Dist, AA );

else
    
    [p1,~] = size(A);
    [p2,~] = size(B);
    if p1~=p2, error('A and B should have the same number of rows'); end
    
    AA = sum(A.^2,1);
    BB = sum(B.^2,1);

    
    Dist = -2*(A'*B);
    Dist = bsxfun( @plus, Dist, AA' );
    Dist = bsxfun( @plus, Dist, BB );
    % fprintf('Discrepancy between the two approaces: %.2e\n', norm(Dist-Dist_Ref,'fro')/norm(Dist_Ref,'fro') );
end
end