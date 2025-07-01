function Ksub = polynomialKernel(X,rowInd,colInd,kernel)
    %% Guassian kernel generator
    % Outputs a submatrix of the Gaussian kernel with variance paramater 
    % gamma for the data rows of X. 
    %
    % usage : 
    %
    % input:
    %
    %  * X : A matrix with n rows (data points) and d columns (features)
    %
    %  * rowInd, colInd : Lists of indices between 1 and n. 
    %
    %  NOTE: colInd can be an empty list, in which case the **diagonal** 
    %  entries of the kernel will be output for the indices in rowInd.
    %  
    %  * gamma : kernel variance parameter
    %
    % output:
    %
    %  * Ksub : Let K(i,j) = (X(i,:)X(j,:)'+c).^d. Then Ksub = 
    %  K(rowInd,colInd). Or if colInd = [] then Ksub = diag(K)(rowInd).
   d = kernel.par(1); c = kernel.par(2); 
    if isempty(colInd)
        % Diagonal elements: K(i,i) = (||x_i||^2 + c)^d
        normsq = sum(X(rowInd,:).^2, 2);
        Ksub = (normsq + c).^d;
    else
        % Off-diagonal blocks: K(i,j) = (x_i'*x_j + c)^d
        innerProd = X(rowInd,:) * X(colInd,:)';
        Ksub = (innerProd + c).^d;
    end
end 