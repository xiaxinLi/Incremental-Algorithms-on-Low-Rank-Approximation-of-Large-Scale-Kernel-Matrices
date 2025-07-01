function [Z,Omega,beta] = Random_Fourier(X,bw,nmbFeatures,Omega,beta)

[n,p] = size(X);

if nargin == 3
    gamma = 1/(bw^2);
    [Omega,beta] = gaussianRandomFeatures(nmbFeatures,p,gamma);
end
Z = transformFeatures(X, Omega, beta );
% K = Z*Z';

end


%% Utilities

function [Omega,beta] = gaussianRandomFeatures(D, p,gamma)   
    % sample random Fourier directions and angles
    Omega = sqrt(2*gamma)*randn(p,D); % RVs defining RFF transform
    beta = rand(1,D)*2*pi; 
end

function [ Z ] = transformFeatures( X, Omega, beta )
    %TRANSFORMFEATURES Transforms data to the random Fourier feature space
    %
    %   Input: 
    %   X - n x p data matrix (each row is a sample) 
    %   Omega - p x D matrix of random Fourier directions (one for each
    %   dimension of a sample x)
    %   beta - 1 x D vector of random angles
    %
    %   Output:
    %   Z - n x D matrix of random Fourier features

    D = size(Omega,2);
    Z = cos(bsxfun(@plus,X*Omega,beta))*sqrt(2/D);
end
