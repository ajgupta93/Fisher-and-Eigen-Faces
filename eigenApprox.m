function [ I ] = eigenApprox( x, W, k )

% output image approximated by k eigenvectors from W
d = 2500;
Wproj = zeros(k, d);
Wproj(1:k, :) = W(1:k, :);

x_proj = Wproj * x;
xnew = Wproj'* x_proj;

I = xnew;

end

