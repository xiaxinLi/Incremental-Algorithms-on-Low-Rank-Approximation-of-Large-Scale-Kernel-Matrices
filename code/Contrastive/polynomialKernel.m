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
    if(isempty(colInd))
        Ksub = ones(length(rowInd),1);
    else
        d = kernel.par(1); c = kernel.par(2);
        X1=X(rowInd,:);
        X2=X(colInd,:);
        Ksub = (X1 * X2' + c).^d; % C: n*m
    end
end 