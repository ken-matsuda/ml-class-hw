function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
thetaLength = length(theta)

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

yGreaterThan1 = -1*y'*log(sigmoid(X*theta));
yLessThan1 = -1*(1-y')*log(1-sigmoid(X*theta));
regTerm = (lambda/(2*m))*sum(theta(2:thetaLength).^2)

J = 1/m*sum(yGreaterThan1+yLessThan1) + regTerm;

%regTermPartDeriv = (lambda/m)*theta

grad = (X'*(sigmoid(X*theta)-y))/m;
grad = [grad(1);grad(2:thetaLength)+((lambda/m)*theta(2:thetaLength))]'



% =============================================================

end
