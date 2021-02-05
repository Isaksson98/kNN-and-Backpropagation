function [ acc ] = calcAccuracy( cM )
% CALCACCURACY Takes a confusion matrix amd calculates the accuracy

% Add your own code here
    acc = 0;
    acc = trace(cM)/sum(sum(cM))*100;
end

