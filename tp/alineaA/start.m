clear all;
close all;
clc
for fd = 1:1

close all;
fprintf('\n\nExecução Nº %d\n', fd);
[input,targets] = binarizedStartData;

net = feedforwardnet(10); % 1 camada oculta com 10 neurônios, respectivamente

net.layers{1}.transferFcn = 'tansig';
% net.layers{2}.transferFcn = 'tansig';
% net.layers{3}.transferFcn = 'tansig';
% net.layers{4}.transferFcn = 'tansig';
% net.layers{5}.transferFcn = 'tansig';
% net.layers{6}.transferFcn = 'tansig';
% net.layers{7}.transferFcn = 'tansig';
% net.layers{8}.transferFcn = 'tansig';
net.layers{end}.transFerFcn = 'purelin';

net.trainFcn = 'trainlm';

net.trainParam.epochs = 1000; % número máximo de épocas de treinamento
%% Divida os dados em conjuntos de treinamento, validação e teste:

net.divideFcn = ''; % divisão aleatória dos dados
%net.divideMode = 'sample'; % divisão por amostra
% net.divideParam.trainRatio = 0.9;
% net.divideParam.valRatio = 0.05;
% net.divideParam.testRatio = 0.05;

%% Treine a rede neural:

%net.trainParam.goal = 0.01; % meta de erro de treinamento
%net.trainParam.showWindow = true; % mostrar a janela de treinamento
[net, tr] = train(net, input, targets); % treinamento da rede neural

out = sim(net, input);
plotconfusion(targets,out);

plotperf(tr)         % Grafico com o desempenho da rede nos 3 conjuntos           

r=0;
for i=1:size(out,2)               % Para cada classificacao  
  [a, b] = max(out(:,i));         %b guarda a linha onde encontrou valor mais alto da saida obtida
  [c, d] = max(targets(:,i));     %d guarda a linha onde encontrou valor mais alto da saida desejada
  if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
  end
end

accuracy = r/size(out,2)*100;
fprintf('Precisao total: %f\n', accuracy)

% Teste a rede neural:
TInput = input(:, tr.testInd);
TTargets = targets(:, tr.testInd);

out = sim(net, TInput);

%erro = perform(net, out,TTargets);
%fprintf('Erro na classificação do conjunto de teste %f\n', erro)

%Calcula e mostra a percentagem de classificacoes corretas no conjunto de teste

r=0;
for i=1:size(tr.testInd,2)               % Para cada classificacao  
  [a, b] = max(out(:,i));                %b guarda a linha onde encontrou valor mais alto da saida obtida
  [c, d] = max(TTargets(:,i));           %d guarda a linha onde encontrou valor mais alto da saida desejada
  if b == d                              % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
  end
end
accuracy = r/size(tr.testInd,2)*100;
fprintf('Precisao teste %f\n', accuracy)

str1 = 'train_nr';
str2 = num2str(fd);
result = strcat(str1, '', str2);
save(result, 'net');

end





