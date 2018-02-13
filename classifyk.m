function [ error ] = classifyk( trainset, trainlabels, testset_num, p , k)

% Given a testset number:
% load the testset
% use p_norm to calculate distance
% and k-NN to classify

% load testset
[testset testlabels]=loadSubset(testset_num);
[num dim] = size(testset);

% iterate
[n d] = size(trainset);

counter = 0;

for x = 1:num
    % initialization
    NN = zeros(2, n);
    min = Inf;
    
    for y = 1:n
        %{
        testset(x, :) = testset(x, : ) - mean(testset(x, :));
        trainset(y, :) = trainset(y, : ) - mean(trainset(y, :));
        %}
        
        diff_img = testset(x, : ) - trainset(y, : );
        diff_img = abs(diff_img);
        diff_img = power(diff_img, p);
        S = sum(diff_img);
        distance = power(S, 1./p);
        
        NN(1, y) = distance;
        NN(2, y) = trainlabels(y);
        if distance < min
            min = distance;
            min_label = trainlabels(y);
        end
           
    end
    
    % find k-NN
    sorted_NN = sortrows(NN',1)';
    
    %disp(sorted_NN(2, :));
    % find majority
    X = sorted_NN(2, 1:k);
    label = mode(X);
    
    %{
    M = zeros(1, 10);
    for i = 1:k
        M(1, sorted_NN(2, i)) = M(1, sorted_NN(2, i)) + 1;
    end
    
    max_value = max(M);
    %label = zeros(1, 10);
    for j = 1:10
        if M(1, j) == max_value
            %label(1, j) = j;
            label = j;
        end
    end
    %}
    % break ties
    
    if label == testlabels(x)
        counter = counter + 1;
        %disp(x);disp(label);
    %else
        %A = [x label min_label testlabels(x)];
        %disp(A);
        %disp(sorted_NN(2, 1:5));
        
    end
end

error = (num - counter) ./ num;

end

