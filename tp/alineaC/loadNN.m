function [] = loadNN(fileName1, fileName2)

close all;
%net = load ('train_nr_61.mat', 'net');
loadedData1 = load(fileName1, 'netNum'); % Load the neural network model
net1 = loadedData1.netNum; % Extract the neural network object from the loaded data

loadedData2 = load(fileName2, 'netOpr'); % Load the neural network model
net2 = loadedData2.netOpr; % Extract the neural network object from the loaded data

[inNumbers, inOperators, trNumbers, trOperators] = binarizedCheckData();

outNum = sim(net1, inNumbers);
figure;
plotconfusion(trNumbers, outNum);

r=0;
for i=1:size(outNum,2)
    [~, b] = max(outNum(:,i));      
    [~, d] = max(trNumbers(:,i)); 
    if b == d 
      r = r+1;
    end
    %fprintf('escolheu %d\n', b);
end

accuracy = r/size(outNum,2);
fprintf('\nNumeros: %d\n',accuracy*100);


outOpr = sim(net2, inOperators);
figure;
plotconfusion(trOperators, outOpr);

r=0;
for i=1:size(outOpr,2)               
    [~, b] = max(outOpr(:,i));      
    [~, d] = max(trOperators(:,i)); 
    if b == d 
      r = r+1;
    end
    %fprintf('escolheu %d\n', b);
end

% o d é o que era suposto, o b é o que a rede escolheu
accuracy = r/size(outOpr,2);
fprintf('Operadores: %d',accuracy*100);

end