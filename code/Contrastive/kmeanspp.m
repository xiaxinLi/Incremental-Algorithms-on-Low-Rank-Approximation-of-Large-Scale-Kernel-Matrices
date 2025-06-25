function [Z,labels] = kmeanspp(X, k,max_iters)
% Input: 
%       X           n*m         列为样本
%       K           scalar      Sample size（聚类种类）
%      max_iters    scalar      迭代次数     
% Output:
%       centroids   n*K         簇质心（选择的样本）
%       labels      1*n         Kernel k-means labels
%
%聚类种类
X=X';
centroids = init_centroids(X, k);

% 迭代更新簇分配和簇质心
for i = 1:max_iters
    % 簇分配
    labels = assign_labels(X, centroids);
    % 更新簇质心
    centroids = update_centroids(X, labels, k);
end
Z=centroids';
% 初始化簇质心函数
function centroids = init_centroids(X, k)
    % 随机选择一个数据点作为第一个质心
    centroids = X(randperm(size(X, 1), 1), :);
    % 选择剩余的质心
    for j = 2:k
        D = pdist2(X, centroids, 'squaredeuclidean');
        D = min(D, [], 2);
        D = D / sum(D);
        centroids(j, :) = X(find(rand < cumsum(D), 1), :);
    end
end

% 簇分配函数
function labels = assign_labels(X, centroids)
    [~, labels] = min(pdist2(X, centroids, 'squaredeuclidean'), [], 2);
end

 % 更新簇质心函数
function centroids = update_centroids(X, labels, K)
    centroids = zeros(K, size(X, 2));
    for z = 1:K
        centroids(z, :) = mean(X(labels == z, :), 1);
    end
end
end