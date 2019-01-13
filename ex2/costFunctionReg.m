function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
Z = X*theta;
H = sigmoid(Z);
LH = log(H);
YLH = y.*LH;
NYLH = -1.*YLH;
OMH = 1.-H;
LOMH = log(OMH);
OMY = 1.-y;
PYLH = OMY.*LOMH;
J = NYLH - PYLH;
J = sum(J);
J = J/m;


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


% You need to return the following variables correctly

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
