function [Z,labels] = kmeanspp(X, k,max_iters)
% Input: 
%       X           n*m         Column is a sample
%       K           scalar      Sample size
%      max_iters    scalar      Maximum number of iterations     
% Output:
%       centroids   n*K         Cluster centroid (sample of choice)
%       labels      1*n         Kernel k-means labels
%

X=X';
centroids = init_centroids(X, k);

% Iteratively update the cluster assignment and cluster centroid
for i = 1:max_iters
    % Cluster assignment
    labels = assign_labels(X, centroids);
    % Update the cluster centroid
    centroids = update_centroids(X, labels, k);
end
Z=centroids';
% Initialize the cluster centroid function
function centroids = init_centroids(X, k)
    % A data point is randomly selected as the first centroid
    centroids = X(randperm(size(X, 1), 1), :);
    % Select the remaining centroids
    for j = 2:k
        D = pdist2(X, centroids, 'squaredeuclidean');
        D = min(D, [], 2);
        D = D / sum(D);
        centroids(j, :) = X(find(rand < cumsum(D), 1), :);
    end
end

% Cluster assignment function
function labels = assign_labels(X, centroids)
    [~, labels] = min(pdist2(X, centroids, 'squaredeuclidean'), [], 2);
end

 % Update the cluster centroid function
function centroids = update_centroids(X, labels, K)
    centroids = zeros(K, size(X, 2));
    for z = 1:K
        centroids(z, :) = mean(X(labels == z, :), 1);
    end
end
end