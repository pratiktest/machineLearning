function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
% number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

%load('ex3data1.mat');

%theta_t = [-2; -1; 1; 2];
%X_t = [ones(5,1) reshape(1:15,5,3)/10];
%y_t = ([1;0;1;0;1] >= 0.5);
%lambda_t = 3;

%theta = theta_t;
%X = X_t;
%y = y_t;
%lambda = lambda_t;
m = length(y)

H = sigmoid(X*theta);

F = -1.*y.*log(H);

S = (1.-y).*log(1.-H);

J = sum(F-S)/m;


TS = theta.^2;
Exclude = TS(1);
TS = sum(TS);
TS = TS-Exclude;
TS = (lambda/(2*m)) * TS;
J = J + TS;

derivative = (X'*(H-y))/m;

grad = derivative;
temp=theta;
temp(1) = 0;
grad = grad + temp*(lambda/m);




% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

%grad = grad(:);

end
