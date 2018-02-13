function [ error ] = classify1( trainset, trainlabels, testset_num, p )

% Given a testset number:
% load the testset
% use p_norm to calculate distance
% and 1-NN to classify

% load testset
[testset testlabels]=loadSubset(testset_num);
%figure
%imshow(drawFaces(testset, 10));
[num dim] = size(testset);

% iterate
[n d] = size(trainset);

counter = 0;

for x = 1:num
    % initialization
    min = Inf;
    label = -1;
    
    for y = 1:n
        
        % using lp-norm formula to find distance by first taking absolute
        % value of the difference betweent the training and testing images.
        diff_img = testset(x, : ) - trainset(y, : );
        diff_img = abs(diff_img);
        diff_img = power(diff_img, p);
        S = sum(diff_img);
        distance = power(S, 1./p);
        
        if distance < min
            min = distance;
            label = trainlabels(y);
        end
    end
    
    if label == testlabels(x)
        counter = counter + 1;
        %disp(x);disp(label);
    end
end

error = (num - counter) ./ num;

end

