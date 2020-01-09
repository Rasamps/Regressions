function a3q3_20051918
% Answer for CISC371, Fall 2019, Assignment #3, Question #3

% Read the data from a CSV file
mtab = readtable('mtls.csv', 'DateLocale', 'de_US');
datamat = table2array(mtab(1:end,2:end));

% Set the dependent column as specified
ycol = 2;

% Problem size
[m, n] = size(datamat);

% Extract the design matrix and the dependent vector
xmat = datamat(:,find((1:n)~=ycol));
yvec = datamat(:,ycol);

% Optionally, standardize the data
std_flag = false;
if std_flag
    xmat = zscore(xmat);
    yvec = zscore(yvec);
end

% Set a large lambda and a feasible lambda
lambda_zero = 0;
lambda = .5;

%Set the flag for least squares or lasso regression.
lsflag = true;
lssflag = false;

% Call the CLS code twice: find OLS, then find CLS
rng('default');
[trainbig testbig] = kfold(xmat, yvec, lsflag, lambda_zero, 5);
rng('default');
[trainsmL testsml] = kfold(xmat, yvec, lsflag, lambda, 5);

% Call the student implementation of k-fold LASSO 
rng('default');
[trainlss testlss] = kfold(xmat, yvec, lssflag, 5);

% Display the results
disp(sprintf('  Using column=%d', ycol));
disp('       OLS    CLS   LASSO             OLS     CLS   LASSO');
disp(sprintf('fit=% 0.3f % 0.3f % 0.3f    pred=% 0.3f % 0.3f % 0.3f', ...
     trainbig, trainsmL, trainlss, testbig, testsml, testlss));

std_flag = true;
if std_flag
    xmat = zscore(xmat);
    yvec = zscore(yvec);
end

% Set a large lambda and a feasible lambda
lambda_zero = 0;
lambda = .5;

%Set the flag for least squares or lasso regression.
lsflag = true;
lssflag = false;

% Call the CLS code twice: find OLS, then find CLS
rng('default');
[trainbig testbig] = kfold(xmat, yvec, lsflag, lambda_zero, 5);
rng('default');
[trainsmL testsml] = kfold(xmat, yvec, lsflag, lambda, 5);

% Call the student implementation of k-fold LASSO 
rng('default');
[trainlss testlss] = kfold(xmat, yvec, lssflag, 5);

% Display the results
disp(sprintf('  Using column=%d', ycol));
disp('       OLS    CLS   LASSO             OLS     CLS   LASSO');
disp(sprintf('fit=% 0.3f % 0.3f % 0.3f    pred=% 0.3f % 0.3f % 0.3f', ...
     trainbig, trainsmL, trainlss, testbig, testsml, testlss));
end

% %
% % STUDENT CODE GOES HERE:
% % copy functions "clskfold", "cls" from previous question;
% % add function "lassokfold" (which uses MATLAB "lasso")
% %

function [rmstrain,rmstest]=kfold(xmat, yvec, mflag, theta, k_in)
% [WCLS,LAMBDA]=CLSKFOLD(XMAT,YVEC,THETA,K) performs a k-fold validation
% of the constrained least squares linear fit of YVEC to XMAT, with
% a solution tolerance of NORM(WCLS)^2<=THETA. See CLS for WCLS details.
% If K is omitted, the default is 5.
%
% INPUTS:
%         XMAT    - MxN data vector
%         YVEC    - Mx1 data vector
%         THETA   - positive scalar, solution threshold
%         K       - positive integer, number of folds to use
%         mflag   - if true performs cls, otherwise performs lasso
%                   regression
% OUTPUTS:
%         RMSFIT  - mean root-mean-square error of the fits for the folds
%         RMSPRED - mean RMS error of predicting the test folds

% Problem size
M = size(xmat, 1);

% Set the number of folds; must be 1<k<M
if nargin >= 5 & ~isempty(k_in)
  k = max(min(round(k_in), M-1), 2);
else
  k = 5;
end

% Create array of computed Lagrange multipliers, for debugging
w_lambdalist = zeros(1,k);

% Randomly assign the data into k folds; discard any remainders
one2M = 1:M;
Mk = floor(M/k);
ndxmat = reshape(randperm(M,Mk*k), k, Mk);
% To compute RMS of fit and prediction, we will sum the variances
vartrain  = 0.0;
vartest = 0.0;

% Process each fold
for ix=1:k
    % %
    % % STUDENT CODE GOES HERE: replace these 5 null-effect lines to
    % % (1) set up the "train" and "test" indexing for "xmat" and "yvec"
    % % (2) use the indexing to set up the "train" and "test" data
    % % (3) compute "w_cls" for the fit data
    % %
    curr = ndxmat(ix,:);
    xmat_train = xmat;
    xmat_train([curr],:) = [];
    yvec_train = yvec;
    yvec_train([curr],:) = []; 
    xmat_test = xmat(curr,:);
    yvec_test = yvec(curr,:);
    if mflag
        wstar = cls(xmat_train,yvec_train,theta);
    else
        [b,fitinfo] = lasso(xmat_train,yvec_train);
        lam = min(fitinfo.MSE);
        wstar = b(:,fitinfo.MSE==lam);
    end
    % From "w_cls", find the variances of the fit and the predicition
    vartrain  = vartrain  + rms(xmat_train*wstar  - yvec_train )^2;
    vartest = vartest + rms(xmat_test*wstar - yvec_test)^2;
end
%disp(sprintf('  ***clskfold for k=%02d: lambda =', k));
%disp(wlamlist);
rmstrain  = sqrt(vartrain/k);
rmstest = sqrt(vartest/k);
end

function [w_cls, lambda] = cls(xmat, yvec, theta)
% [WCLS,LAMBDA]=CLS(XMAT,YVEC,THETA) solves constrained
% least squares of a linear regression of YVEC to XMAT, with
% a solution tolerance of NORM(WCLS)^2<=THETA. WCLS is
% the constrained weight vector and LAMBDA is the Lagrange
% multiplier for the solution
%
% INPUTS:
%         XMAT   - MxN design matrix
%         YVEC   - Mx1 data vector
%         THETA  - positive scalar, solution threshold
% OUTPUTS:
%         WCLS   - solution coefficients
%         LAMBDA - Lagrange coefficient

% Return immediately if the threshold is invalid
if theta<0
    w_cls = [];
    lambda = [];
    return;
end

% Set up the problem as xmat*w=yvec
Im = eye(size(xmat, 2));
wfun =@(lval) inv(xmat'*xmat + lval*Im)*xmat'*yvec;
gfun =@(lval) wfun(lval)'*wfun(lval) - theta;

% OLS solution: use pseudo-inverse for ill conditioned matrix
if cond(xmat)<1e+8
    wls = xmat\yvec;
else
    wls = pinv(xmat)*yvec;
end

% The OLS solution is used if it is within the user's threshold
if norm(wls)^2<= theta | theta<=0
    w_cls = wls;
    lambda = 0;
else
    lambda = fzero(gfun, 1);
    % CLS is a simple closed-form solution
    w_cls = wfun(lambda);
end
end
