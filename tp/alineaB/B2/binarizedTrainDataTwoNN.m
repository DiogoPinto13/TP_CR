function [inNumbers, inOperators, outNumbers, outOperators] =  binarizedTrainDataTwoNN()

IMG_RES = [30 30];

DataPath = ["0","1","2","3","4","5","6","7","8","9","add","div","mul","sub"];
%% Ler e redimensionar as imagens e preparar os targets

binaryMatrixNumbers = zeros(IMG_RES(1) * IMG_RES(2), 50*10);
binaryMatrixOperators = zeros(IMG_RES(1) * IMG_RES(2), 50*4);
count = 1;

for i=1:10
    for j=1:50
        img = imread(sprintf('..\\..\\NN datasets\\train1\\%s\\%d.png', DataPath(i), j));
        %img = rgb2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrixNumbers(:, count) = reshape(binarizedImg, 1, [])';
        count=count+1;
    end
end

count = 1;
for i=11:14
    for j=1:50
        img = imread(sprintf('NN datasets\\train1\\%s\\%d.png', DataPath(i), j));
        %img = rgb2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrixOperators(:, count) = reshape(binarizedImg, 1, [])';
        count=count+1;
    end
end

%% targets:
targetNumbers = zeros(10, 10*50);
targetOperators = zeros(4, 4*50);
starterVar = 0;
for s=1:10 %digitos e operadores
    for i = 1:50
        targetNumbers(s, (starterVar+1):s*50) = 1;
        starterVar = 50 * s;
    end
end

starterVar = 0;
for s=1:4 %digitos e operadores
    for i = 1:50
        targetOperators(s, (starterVar+1):s*50) = 1;
        starterVar = 50 * s;
    end
end


inNumbers = binaryMatrixNumbers;
inOperators = binaryMatrixOperators;
outNumbers = targetNumbers;
outOperators = targetOperators;
end