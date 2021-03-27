
acc_limit = [98, 99, 99, 96];
k_iterations = 100;
result = zeros(k_iterations,4);
for i = 1:4
    [X, D, L] = loadDataSet( i );
    
    n = 3; % n-fold paramter
    
    numBins = n;                    % Number of bins you want to devide your data into
    numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
    selectAtRandom = true;          % true = select samples at random, false = select the first features

    [XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

    
    acc = zeros(k_iterations, n);
    for j = 1:n
        
        %seperate data into train/test
        A = [1:n];
        B = [j];
        arr = setdiff(A,B);
        
        XTrain = combineBins(XBins, arr);
        LTrain = combineBins(LBins, arr);
        XTest  = XBins{j};
        LTest  = LBins{j};
        
        %classify
        acc(:,j) = cross_valid(XTrain, XTest, LTrain, LTest, k_iterations);
    end 
    result(:,i) = mean(acc,2);
    figure(i);
    
    plot(result(2:k_iterations,i));
    title(i);
    xlabel('k');
    ylabel('accuracy');
    hline = refline([0 acc_limit(i)]);
    hline.Color = 'r';
end



function acc = cross_valid(XTrain, XTest,LTrain, LTest, iterations)
    
    acc  = zeros(iterations,1);
    
    for k = 2:iterations
        LPred = kNN(XTest, k, XTrain, LTrain);
        cM = calcConfusionMatrix(LPred, LTest);
        acc(k) = calcAccuracy(cM);
    end    
end