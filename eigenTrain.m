function [W,mu] = eigenTrain(trainset,k)

%% find mu first
n = 70;
d = 2500;
sum = zeros(d, 1);

for i = 1:n
    %trainset(i, :) = trainset(i, : ) - mean(trainset(i, :));
    sum = sum + trainset(i, :)';
end

mu = sum./ n;

%% create data matrix D
D = zeros(d, n);

for i = 1:n
    D(:, i) = trainset(i, :)' - mu;
end

%% perform SVD to create covariance mtx
[U, S, V] = svd(D);

Covar = (n - 1) * U;

% normalization
for i = 1:n
    Covar(:, i) = Covar(:, i)/norm(Covar(:, i), 2);
end

%% create W of k eigenvectors
W = zeros(k, d);

for i = 1:k
    W(i, :) = Covar(:, i)';
end

end

