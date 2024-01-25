function [inNumbers, inOperators, outNumbers, outOperators] =  binarizedCheckData()

numImagens = 3;
IMG_RES = [25 25];

DataPath = ["0","1","2","3","4","5","6","7","8","9","add","div","mul","sub"];
%% Ler e redimensionar as imagens e preparar os targets

binaryMatrixNumbers = zeros(IMG_RES(1) * IMG_RES(2), numImagens*10);
binaryMatrixOperators = zeros(IMG_RES(1) * IMG_RES(2), numImagens*4);
count = 1;

for i=1:10
    for j=1:numImagens
        img = imread(sprintf('..\\NN datasets\\personalDataSet\\%s\\%d.png', DataPath(i), j));
        %img = rgb2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrixNumbers(:, count) = reshape(binarizedImg, 1, [])';
        count=count+1;
    end
end

count = 1;
for i=11:14
    for j=1:numImagens
        img = imread(sprintf('NN datasets\\personalDataSet\\%s\\%d.png', DataPath(i), j));
        %img = rgb2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrixOperators(:, count) = reshape(binarizedImg, 1, [])';
        count=count+1;
    end
end

%% targets:
targetNumbers = zeros(10, 10*numImagens);
targetOperators = zeros(4, 4*numImagens);
starterVar = 0;
for s=1:10 %digitos e operadores
    for i = 1:numImagens
        targetNumbers(s, (starterVar+1):s*numImagens) = 1;
        starterVar = numImagens * s;
    end
end

starterVar = 0;
for s=1:4 %digitos e operadores
    for i = 1:numImagens
        targetOperators(s, (starterVar+1):s*numImagens) = 1;
        starterVar = numImagens * s;
    end
end

%save('trNum.mat', 'targetNumbers');
%save('trOpr.mat', 'targetOperators');

inNumbers = binaryMatrixNumbers;
inOperators = binaryMatrixOperators;
outNumbers = targetNumbers;
outOperators = targetOperators;
end

