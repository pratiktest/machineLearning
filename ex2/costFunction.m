function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
m = length(y);
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

% You need to return the following variables correctly
G = H-y;
G = G.*X;
grad = G;
S = [sum(G(:,1));sum(G(:,2));sum(G(:,3))];
S = S./m;
grad = S;



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
