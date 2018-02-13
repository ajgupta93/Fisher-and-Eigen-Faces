function testlabels = eigenTest(trainset,trainlabels,testset,W,mu,k)

[N d] = size(trainset);
[M d] = size(testset);

%figure
%imshow(drawFaces(testset, 10));
testlabels = zeros(M, 1);

for x = 1:M
    min = Inf;
    label = -1;
    test = eigenProjection(testset(x, :)', W, k);
    
    for y = 1:N
        train = eigenProjection(trainset(y, :)', W, k);
        diff = test(:, 1) - train(:, 1);
        diff = abs(diff);
        S = sum(diff);
        distance = power(S, 1./2);
        
        if distance < min
            min = distance;
            label = trainlabels(y);
        %elseif distance == min
        %    break
        end        
    end
    
    testlabels(x, 1) = label;

end



end

