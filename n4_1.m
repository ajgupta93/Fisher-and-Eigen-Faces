[trainset trainlabels]=loadSubset(0);
[testset testlabels]=loadSubset(1);

figure
imshow(drawFaces(trainset, 7));
figure
imshow(drawFaces(testset, 12));

%% 1-NN
disp('1-NN');
for i = 1:4
    error = classify1( trainset, trainlabels, i, 3 );
    disp(error);
end

%% 3-NN
disp('3-NN');
for i = 1:4
    error = classifyk( trainset, trainlabels, i, 3, 3 );
    disp(error);
end

%% 5-NN
disp('5-NN');
for i = 1:4
    error = classifyk( trainset, trainlabels, i, 3, 5 );
    disp(error);
end