function a3q1_20051918
% Answer for CISC371, Fall 2019, Assignment #3, Question #1

% Acquire the instructor's data
[xvec yvec] = clsdata;

% Append 1's to create the design matrix
xmat = [xvec ones(size(xvec))];

% Compute the ordinary least squares from the normal equation
w_ols = inv(xmat'*xmat)*xmat'*yvec

% Compute the constrained least squares solution
theta = 8;
[w_cls lambda] = cls(xmat, yvec, theta)

% %
% % PLOT: data and OLS fit
% %
plot(xvec, yvec, 'k*', xvec, polyval(w_ols, xvec), 'r-', ...
    'LineWidth', 1.5, 'MarkerSize', 8);
axisadjust(1.1);
xlabel('\boldmath$w_1$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('\boldmath$w_2$', 'Interpreter', 'latex', 'FontSize', 14);
tstring = strcat('\bf{}OLS fit: $\boldmath{}\|\vec{w}\|^2 = ', ...
    sprintf('%0.2f', norm(w_ols)^2), '$');
title(tstring, 'Interpreter', 'latex', 'FontSize', 14);

% %
% % PLOT: CLS fit
% %

%figure(2);
plot(xvec, yvec, 'k*', xvec, polyval(w_ols, xvec), 'r-', ...
    'LineWidth', 1.5, 'MarkerSize', 8);
axisadjust(1.1);
hold on;
plot(xvec, polyval(w_cls, xvec), 'b-', ...
    'LineWidth', 1.5);
hold off;
tstring = strcat('\bf{}CLS fit: $\boldmath{}\|\vec{w}\|^2 = ', ...
    sprintf('%0.2f', norm(w_cls)^2), '$');
title(tstring, 'Interpreter', 'latex', 'FontSize', 14);

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

% Set up the problem as Aw=b
b = yvec;
Im = eye(size(xmat, 2));
wfun =@(l) inv(xmat'*xmat + l*Im)*xmat'*b;
gfun =@(l) wfun(l)'*wfun(l) - theta;

% OLS solution: use pseudo-inverse for ill conditioned matrix
if cond(xmat)<1e+8
    wls = xmat\b;
else
    wls = pinv(xmat)*b;
end

% The OLS solution is used if it is within the user's threshold
if norm(wls)^2<= theta | theta==0
    w_cls = wls;
    lambda = 0;
else
    %
    % STUDENT CODE GOES HERE: can use "fzero" to estimate lambda,
    % here the multiplier and solution vector are zero
    lambda = fzero(gfun, 1);
    % CLS solution is a simple closed form
    w_cls = wfun(lambda); %Replace the zero solution vector with wfun evaluated at the optimal lambda value.
end
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
function axisadjust(axisexpand)
% AXISADJUST(AXISEXPAND) multiplies the current plot
% ranges by AXISEXPAND.  To increase by 5%, use 1.05
%
% INPUTS:
%         AXISEXPAND - positive scalar multiplier
% OUTPUTS:
%         none
% SIDE EFFECTS:
%         Changes the current plot axis

axvec = axis();
axwdth = (axvec(2) - axvec(1))/2;
axhght = (axvec(4) - axvec(3))/2;
axmidw = mean(axvec([1 2]));
axmidh = mean(axvec([3 4]));
axis([axmidw-axisexpand*axwdth , axmidw+axisexpand*axwdth , ...
      axmidh-axisexpand*axhght , axmidh+axisexpand*axhght]);
end
