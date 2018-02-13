[trainset trainlabels]=loadSubset(0);

figure
imshow(drawFaces(trainset, 7));

%% (1) & (2): Test error & Difference between sets
% As the light moves away, the error rate increases
for i = 1:4
    error = classify1( trainset, trainlabels, i, 1 );
    disp(error);
end

%% (5): Norm differences
for i = 1:4
    error = classify1( trainset, trainlabels, i, 3 );
    disp(error);
end
