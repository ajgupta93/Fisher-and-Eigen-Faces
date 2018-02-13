%%6.1
[trainset trainlabels] = loadSubset(0);
c = 10;

%% 6.1
[W, mu] = fisherTrain(trainset, trainlabels, c);

%% 6.2
figure
imshow(mat2gray(drawFaces(W, 1)));

%% 6.3
errors = zeros(4, 9);
for set = 1:4
    for k = 1:9
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
x = 1:1:9
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

%% Approximation
figure
for i = 1:10
    set = zeros(10, 2500);
    subplot(1, 10, i)
    for k = 1:9
        x = trainset(7*i-1, :)';
        I = eigenApprox(x, W, k);
        set(k, :) = I + mu;
    end
    set(10, :) = trainset(7*i, :)' + mu;
    imshow(mat2gray(drawFaces(set, 1)));
end