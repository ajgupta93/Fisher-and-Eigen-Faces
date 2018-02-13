function [ xnew ] = eigenProjection( x, W, k )

d = 2500;
Wproj = zeros(k, d);
Wproj(1:k, :) = W(1:k, :);

x_proj = Wproj * x;
xnew = Wproj'* x_proj;


end

