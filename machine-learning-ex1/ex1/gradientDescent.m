function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%alpha = 0.01
%J = 6.6929
%alpha = 0.001
%J = 6.4512
%alpha = 0.0003
%J = 6.4298
%alpha = 0.0002
%J = 6.4451
%alpha = 0.0001
%J = 7.7351

    %J = sum((((X*theta) - y).^2))/(2*m)

    
    alpha = .01;
    hypothesis = X*theta;
    errors = hypothesis - y;
    deltaTheta = alpha * ((transpose(X) * errors)/m)
    theta = theta - deltaTheta

    % ==============ex1==============================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
