function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to X to get the initial activation
a1 = [ones(m,1),X];

% calculate the 2nd layer input and activation units.  add ones to the 
% hidden layer
z2 = a1*Theta1'
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];

% calculate the hypothesis
z3 = a2*Theta2';
hxk = sigmoid(z3);
a3 = hxk;

% make the y matrix
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% next calculate the cost function
jGreaterThanEqualToZero = y_matrix.*log(hxk);
jLessThanZero = (1-y_matrix).*log(1-hxk);

% clearest way to get Theta without the bias is to index to size, but
% for the columns.  There must be a clever Matlab notation for this.
Theta1WithoutTheBias = Theta1(1:size(Theta1,1),2:size(Theta1,2));
Theta2WithoutTheBias = Theta2(1:size(Theta2,1),2:size(Theta2,2));

% double sum of the parameters squared
theta1Calc = sum(sum(Theta1WithoutTheBias.^2));
theta2Calc = sum(sum(Theta2WithoutTheBias.^2));

% just for the sake of sanity, calculate the numerator and denominator
% and then calculate the regularization parameter
numerator = lambda*(theta1Calc+theta2Calc);
denominator = 2*m;
regParam = numerator/denominator;

% double sum of the cost function + the regularization parameter
J = -1*(sum(sum(jGreaterThanEqualToZero + jLessThanZero)))/m + regParam;

% -------------------------------------------------------------
% calculate the back propagation
% -------------------------------------------------------------

% 1. get the delta values
d3 = a3 - y_matrix;
d2 =(d3*Theta2WithoutTheBias).*sigmoidGradient(z2);

Delta1 = d2'*a1;
Delta2 = d3'*a2;

r1num = size(Theta1WithoutTheBias,1);
r2num = size(Theta2WithoutTheBias,1);

reg1 = [zeros(r1num,1) (lambda/m).*Theta1WithoutTheBias];
reg2 = [zeros(r2num,1) (lambda/m).*Theta2WithoutTheBias];

Theta1_grad = Delta1/m+reg1;
Theta2_grad = Delta2/m+reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
