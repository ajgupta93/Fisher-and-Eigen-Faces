function [W, mu] = fisherTrain(trainset, trainlabels, c)

[N, d] = size(trainset);

% (1) Find W_pca
k = N - c;
[W_pca, m] = eigenTrain(trainset, k);

% (2) Project the training data onto dim(N-c)
train_proj = W_pca * trainset';

% (3-1) Compute SB
SB = zeros(k, k);

% compute mu
mu = zeros(k, 1);
for i = 1:N
    mu = mu + train_proj(:, i);
end
mu = mu ./ N;

for j = 1:c
    % compute SB_i
    chi_i = 7;
    sum_i = zeros(k, 1);
    
    for i = 1:7
        sum_i = sum_i + train_proj(:, i + (j-1)*7);
    end
    
    mu_i = sum_i ./ chi_i;
    SB_i = chi_i * (mu_i - mu) * (mu_i - mu)';
    SB = SB + SB_i;
end

% (3-2) Compute SW
SW = zeros(k, k);

for j = 1:c
    % compute SW_i
    chi_i = 7;
    sum_i = zeros(k, 1);
    
    for i = 1:7
        sum_i = sum_i + train_proj(:, i + (j-1)*7);
    end
    
    mu_i = sum_i ./ chi_i;
    
    for i = 1:7
        x_k = train_proj(:, i + (j-1)*7)
        SW_i = (x_k - mu_i) * (x_k - mu_i)';
        SW = SW + SW_i;
    end
end

% (4) Compute W_FLD
%[V,D] = eig(inv(SW) * SB);

[W_FLD, D] = eig(SB, SW);

% sort
[D, i] = sort(diag(D), 'descend');
W_FLD = W_FLD(:,i);

% truncate
W_FLD = W_FLD(:,1:c-1);

% transpose
W_FLD = W_FLD';

% (5) Find W
W = W_FLD*W_pca;

mu = m;

end

