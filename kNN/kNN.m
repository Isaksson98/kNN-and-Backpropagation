function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);

%Calculate euclidean distance between all points and stores in matrix
Eucl_distance = pdist2(XTrain,X);
   
%Sorts the distances in ascending order
[sortedVals,indexes] = sort(Eucl_distance);


%Picks out k-nearest neighbours
k_nearest = indexes(1:k,:);


%Get label of neighbours
labels = LTrain(k_nearest);

%Count most common neighbour
if k == 1
    LPred = labels;
else
    LPred = mode(labels);
    LPred = transpose(LPred);
end


end

