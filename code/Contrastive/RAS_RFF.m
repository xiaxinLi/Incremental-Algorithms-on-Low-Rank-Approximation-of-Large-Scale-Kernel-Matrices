%the random adaptive sampling method 

function [idS] = RAS_RFF(X,bw,c,t,epsilon,lambda,nb_FF,updating)

%
rng shuffle;

% Initialize
N = size(X,1);N1 = size(X,2);

% STEP 1: Approximate Leverage scores using RANDOM FOURIER FEATURES
B = Random_Fourier(X,bw,nb_FF);
b = chol(B'*B+N*lambda*eye(nb_FF));
F = b'\B'; % P = F'*F

%idS = zeros(N,1);
idS = RASv0(F,c,t,epsilon,updating);
nmbSamples = nnz(idS);
idS = idS(1:nmbSamples);
idS(idS>N1) = N1;
end

%% 1¡¢Random_Fourierº¯Êý
function [Z,Omega,beta] = Random_Fourier(X,bw,nmbFeatures,Omega,beta)

[n,p] = size(X);

if nargin == 3
    gamma = 1/(bw^2);
    [Omega,beta] = gaussianRandomFeatures(nmbFeatures,p,gamma);
end
Z = transformFeatures(X, Omega, beta );
% K = Z*Z';

end
% Utilities
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

%% 2¡¢RASv0º¯Êý
% P = F'*F

function idS = RASv0(F,c,t,epsilon,updating)

% Dimensions
N = size(F,2);

% Initialize
idS = zeros(N,1); 
weights = zeros(N,1); 

R = 0;

% leverage Score
lev_score = (sum(F.*F,1))';

%Number of sampled landmarks
nmbSamples = 0;

for i = 1:N 

    % First landmark
    if nmbSamples == 0
                
        % probability of sampling
        p_i = min(c*(1/epsilon)*(1+t)*lev_score(i),1);

        % Projection matrix
        if rand < p_i
            nmbSamples = 1;
            idS(nmbSamples) = i;
            weights(nmbSamples) = 1/sqrt(p_i);
            R(1,1) = 1/( lev_score(i) + epsilon./(weights(1:nmbSamples))^2 );
        end 
        
    else
        
        % Calculate leverage score 
        PS_i = F(:,i)'*F(:,idS(1:nmbSamples));
       
        temp = max( 0 ,lev_score(i) - sum((PS_i*R).*PS_i,2) );
        p_i = min( 1 , c*(1/epsilon)*(1+t)*temp );        

          
        % Projection matrix
        if rand < p_i

            nmbSamples = nmbSamples + 1;
            idS(nmbSamples) = i;
            weights(nmbSamples) = 1/sqrt(p_i);
            
            if updating
                newA = F(:,idS(1:nmbSamples))'*F(:,i);
                newA(end,end) = newA(end,end)+ epsilon*weights(nmbSamples).^(-2);
                R = updateLS(R,newA);
            else
                SPS = F(:,idS(1:nmbSamples))'*F(:,idS(1:nmbSamples));
                R = (SPS + diag(epsilon*weights(1:nmbSamples).^(-2)))\eye(nmbSamples);
            end 
            
        end      
    end
end
end
     

function K_inv = updateLS(K_inv,newA)

% Init
a = newA(1:end-1);
alpha = newA(end);

% Update K_inv 
B1 = K_inv + ((K_inv*a)*(a'*K_inv))/(alpha - a'*K_inv*a);
B2 = -K_inv*a/(alpha - a'*K_inv*a);
B3 = -a'*K_inv/(alpha - a'*K_inv*a);
B4 = 1/(alpha-a'*K_inv*a);
K_inv = [B1,B2;B3,B4];
end


 

