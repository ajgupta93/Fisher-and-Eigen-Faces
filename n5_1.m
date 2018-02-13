% load data
[trainset, trainlabels]=loadSubset(0);

% 5.7

figure
for i = 1:4
    [s, l] = loadSubset(i);
    subplot(1, 4, i)
    imshow(drawFaces(s(1, :), 1));
    
end

figure
imshow(drawFaces(trainset, 7));

%% 5.1 get eigenvectors from training set
[W, mu] = eigenTrain(trainset, 20);

%% 5.2 draw eigenfaces
figure
imshow(mat2gray(drawFaces(W, 2)));

%% 5.3 purpose of PCA and what does it achieve?

%% 5.4 approximation
figure
for i = 1:10
    set = zeros(11, 2500);
    subplot(1, 10, i)
    for k = 1:10
        x = trainset(7*i-1, :)';
        I = eigenApprox(x, W, k);
        set(k, :) = I + mu;
    end
    set(11, :) = trainset(7*i, :)' + mu;
    imshow(mat2gray(drawFaces(set, 1)));
end

%% 5.5

errors = zeros(4, 20);
for set = 1:4
    for k = 1:20
        [testset, testlabels]=loadSubset(set);
        labels = eigenTest(trainset,trainlabels,testset,W,mu,k);

        % error rate
        [M d] = size(testlabels);
        counter = 0;
        for i = 1:M
            if labels(i, 1) ~= testlabels(i, 1)
                counter = counter + 1;
            end
        end
        error = counter ./ M;
        errors(set, k) = error;
    end
end

% plot
figure
x = 1:1:20
for set = 1:4
    hold on
    y = errors(set, x);
    plot(x, y, '-o')
    hold off
end
title('Error rate using top k eigenvectors')
xlabel('k')
ylabel('error rate')
legend('set1', 'set2', 'set3', 'set4');

%% 5.6 variation in eigenvectors

[W, mu] = eigenTrain(trainset, 24);
Wnew = zeros(20, 2500);
Wnew(1:20, :) = W(5:24, :);


errors = zeros(4, 20);
for set = 1:4
    for k = 1:20
        [testset, testlabels]=loadSubset(set);
        labels = eigenTest(trainset,trainlabels,testset,Wnew,mu,k);

        % error rate
        [M d] = size(testlabels);
        counter = 0;
        for i = 1:M
            if labels(i, 1) ~= testlabels(i, 1)
                counter = counter + 1;
            end
        end
        error = counter ./ M;
        errors(set, k) = error;
    end
end

% plot
figure
x = 1:1:20
for set = 1:4
    hold on
    y = errors(set, x);
    plot(x, y, '-o')
    hold off
end
title('Error rate using k eigenvectors starting from 5th')
xlabel('k')
ylabel('error rate')
legend('set1', 'set2', 'set3', 'set4');
