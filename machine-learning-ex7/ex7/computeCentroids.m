function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

indexCount = zeros(K,1);

for i=1:K
    % find indices of elements that match centroid i
    indicesOfTrainingExamplesAssignedToCentroid = find(idx==i);
    % then extract those elements into new matrix
    trainingExamplesAssignedToCentroid = X(indicesOfTrainingExamplesAssignedToCentroid,:);
    % calculate the average
    centroids(i,:) = sum(trainingExamplesAssignedToCentroid)/size(trainingExamplesAssignedToCentroid,1);
end

%for i=1:m
    % get the number so far
%    numAssignedTrainingExamples = indexCount(idx(i))+1;
    % multiply by the current count
%    centroid = centroids(idx(i),:) * (numAssignedTrainingExamples-1);
    % add the training example
%    centroid = centroid + X(i,:);
    % divide by the new number of assigned training examples
%    centroid = centroid / numAssignedTrainingExamples;
%    centroids(idx(i),:) = centroid;
%    indexCount(idx(i)) = numAssignedTrainingExamples;
%end






% =============================================================


end

