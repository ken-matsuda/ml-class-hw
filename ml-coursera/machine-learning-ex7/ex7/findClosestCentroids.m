function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%numTrainingExamples = size(X,1);
%numCentroids = size(centroids,1);
%shortestDistance = 0;

%numDimensions = size(X,2);

%for i = 1:numTrainingExamples
%    for j = 1:numCentroids
%        distance = norm(X(i,:) - centroids(j,:));
%        if j==1
%            shortestDistance = distance;
%            idx(i)=j;
%        else 
%            if (distance < shortestDistance)
%                shortestDistance = distance;
%                idx(i)=j;            
%            end
%        end
%    end
%end

m = size(X,1);
diffs = zeros(m,K);

for j=1:K
    % subtract each centroid from X and generate a new matrix
    temp = bsxfun(@minus, X, centroids(j,:));
    % square and sum on the 2nd dimension to get diffs
    diffs(:,j) = sum(temp.^2,2);
end

% get the index of the min of the diffs on the 2nd dimension
[Y,idx] = min(diffs,[],2);

% =============================================================

end

