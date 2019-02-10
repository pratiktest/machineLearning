function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


H = X*theta;
D = H - y;
D = D .^ 2;
J = sum(D);
J = J/(2*m);

TS = theta.^2;
Exclude = TS(1);
TS = sum(TS);
TS = TS-Exclude;
TS = (lambda/(2*m)) * TS;
J = J + TS;

Delta0 = (H-y).*X(:,1);
Delta0 = sum(Delta0);
Delta0 = Delta0/m;

sizeX = size(X)(2);
X = X(:,2:sizeX);
DeltaR = (H-y).*X;
DeltaR = sum(DeltaR);
DeltaR = DeltaR/m;
DeltaR;
ThetaL = theta(2:sizeX);
C = (ThetaL*lambda)/m;
C = C';
DeltaR = DeltaR+C;

grad = [Delta0, DeltaR];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
