function a3q2_20051918
% Answer for CISC371, Fall 2019, Assignment #3, Question #2

% Acquire the instructor's data
[xvec yvec] = clsdata;

% Append 1's to create the design matrix
xmat = [xvec ones(size(xvec))];

% Set the constraint value as specified in the assignment
theta = 8;

% Compute 10 sets of 5-fold validation for ordinary LS and CLS;
% OLS is done using theta=0 as a constraint
vval = 10;
k = 5;
ofvec = zeros(vval, 1);
opvec = zeros(vval, 1);
cfvec = zeros(vval, 1);
cpvec = zeros(vval, 1);

% Set the random number generator to duplicate the instructor's method
rng('default');
%
% STUDENT CODE GOES HERE: compute OLS statistics
%
for tx=1:10
    [rmstrain,rmstest] = clskfold(xmat,yvec,0,5);
    ofvec(tx) = rmstrain;
    opvec(tx) = rmstest;
end

% Set the random number generator to duplicate the instructor's method
rng('default');
% %
% % STUDENT CODE GOES HERE: compute CLS statistics
% %
for tx=1:10
    [rmstrain,rmstest] = clskfold(xmat,yvec,theta,5);
    cfvec(tx) = rmstrain;
    cpvec(tx) = rmstest;
end
disp('The mean and standard deviation of the OLS training data is...')
mean(ofvec)
std(ofvec)
disp('The mean and standard deviation of the OLS testing data is...')
mean(opvec)
std(opvec)
disp('The mean and standard deviation of the CLS training data is...')
mean(cfvec)
std(cfvec)
disp('The mean and standard deviation of the CLS testing data is...')
mean(cpvec)
std(cpvec)
end

function [xvec, yvec] = clsdata
% [XVEC,YVEC]=CLSDATA creates a small data set for testing
% linear regression. XVEC contains equally spaced points.
% YVEC is mainly an affine transformation of XVEC, with the first
% and last values deviated.
%
% INPUTS:
%         none
% OUTPUTS:
%         XVEC - Mx1 vector of independent data
%         YVEC - Mx1 vector of   dependent data

% X values are equally spaced
xvec = linspace(0, 9, 10)';

% Y are linear, deviating first and last
ylin = exp(1)*xvec + pi;
yvec = [(ylin(1) - 5) ; ylin(2:end-1) ; ylin(end) + 3];
end

function [rmstrain,rmstest]=clskfold(xmat, yvec, theta, k_in)
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
% OUTPUTS:
%         RMSFIT  - mean root-mean-square error of the fits for the folds
%         RMSPRED - mean RMS error of predicting the test folds

% Problem size
M = size(xmat, 1);

% Set the number of folds; must be 1<k<M
if nargin >= 4 & ~isempty(k_in)
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
    xmat_train(curr,:) = [];
    yvec_train = yvec;
    yvec_train(curr,:) = []; 
    xmat_test = xmat(curr,:);
    yvec_test = yvec(curr,:);
    w_cls = cls(xmat_train,yvec_train,theta);
    
    % From "w_cls", find the variances of the fit and the predicition
    vartrain  = vartrain  + rms(xmat_train*w_cls  - yvec_train )^2;
    vartest = vartest + rms(xmat_test*w_cls - yvec_test)^2;
end
%disp(sprintf('  ***clskfold for k=%02d: lambda =', k));
%disp(wlamlist);
rmstrain  = sqrt(vartrain/k);
rmstest = sqrt(vartest/k);
end

function [w_cls, lambda] = cls(xmat, yvec,theta)
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
