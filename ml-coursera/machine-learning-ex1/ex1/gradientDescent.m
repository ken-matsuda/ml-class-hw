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
    %hx = X*theta;
    %prod = X'*(hx-y); % forgot to subtract
    %learningTerm = (alpha/m).*prod;
    %theta = theta-learningTerm;
    hx = X*theta;
    theta = theta - (alpha/m).*(X'*(hx-y));

    % ============================================================

    % Save the cost J in every iteration    
    J = computeCost(X, y, theta);
    J
    J_history(iter) = J;

end

end
