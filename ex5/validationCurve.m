function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

m = size(X,1);
n = size(Xval, 1);

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    thetaT = trainLinearReg(X,y, lambda);
    H = X*thetaT;
    D = H - y;
    D = D .^ 2;
    J = sum(D);
    J = J/(2*m);
    error_train(i) = J;

    Hcv = Xval*thetaT;
    Dcv = Hcv - yval;
    Dcv = Dcv .^ 2;
    Jcv = sum(Dcv);
    Jcv = Jcv/(2*n);
    error_val(i) = Jcv;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%

%










% =========================================================================

end
