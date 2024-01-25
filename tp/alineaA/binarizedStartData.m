function [in, target] =  binarizedStartData()

IMG_RES = [25 25]; % de 150x150 passa a 25x25

DataPath = ["0","1","2","3","4","5","6","7","8","9","add","mul","sub","div"];
%% Ler e redimensionar as imagens e preparar os targets

binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), 5*14);
targetMatrix = [];
count = 1;

for i=1:14
    for j=1:5
        img = imread(sprintf('..\\NN datasets\\start\\%s\\%d.png', DataPath(i), j));
        %img = rgb2gray(img);
        img = imresize(img, IMG_RES);
        binarizedImg = imbinarize(img);
        binaryMatrix(:, count) = reshape(binarizedImg, 1, [])';
        count=count+1;
    end
    %count = 1;
end

%% targets:
target = zeros(14, 14*5);
starterVar = 0;
for s=1:14 %digitos e operadores
    for i = 1:5
        target(s, (starterVar+1):s*5) = 1;
        starterVar = 5 * s;
    end
end


in = binaryMatrix;
end