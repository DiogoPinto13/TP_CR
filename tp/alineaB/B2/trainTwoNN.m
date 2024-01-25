clear all;
close all;
clc

for iteracao = 1:1
close all;


%[input,targets] = binarizedTrainData;
% disp(iteracao);
fprintf('\n\nExecução Nº %d\n', iteracao);
[inNumbers, inOperators, trNumbers, trOperators] = binarizedTrainDataTwoNN();


netNum = feedforwardnet(20); % 1 camada oculta com 20 neurónios, respectivamente

netNum.layers{1}.transferFcn = 'tansig';
% netNum.layers{2}.transferFcn = 'tansig';
% netNum.layers{3}.transferFcn = 'tansig';
% netNum.layers{4}.transferFcn = 'tansig';
% netNum.layers{5}.transferFcn = 'tansig';
% netNum.layers{6}.transferFcn = 'tansig';
% netNum.layers{7}.transferFcn = 'tansig';
% netNum.layers{8}.transferFcn = 'tansig';
% netNum.layers{9}.transferFcn = 'tansig';
% netNum.layers{10}.transferFcn = 'tansig';

netNum.layers{end}.transFerFcn = 'purelin';

netNum.trainFcn = 'trainlm';

netNum.trainParam.epochs = 50; % número máximo de épocas de treinamento

%% Divida os dados em conjuntos de treinamento, validação e teste:

%netNum.divideFcn = ''; % divisão aleatória dos dados
%netNum.divideMode = 'sample'; % divisão por amostra
netNum.divideParam.trainRatio = 0.7;
netNum.divideParam.valRatio = 0.15;
netNum.divideParam.testRatio = 0.15;

%% Operators
netOpr = feedforwardnet(20); % 1 camada oculta com 20 neurónios, respectivamente

netOpr.layers{1}.transferFcn = 'tansig';
% netOpr.layers{2}.transferFcn = 'tansig';
% netOpr.layers{3}.transferFcn = 'tansig';
% netOpr.layers{4}.transferFcn = 'tansig';
% netOpr.layers{5}.transferFcn = 'tansig';
% netOpr.layers{6}.transferFcn = 'tansig';
% netOpr.layers{7}.transferFcn = 'tansig';
% netOpr.layers{8}.transferFcn = 'tansig';
% netOpr.layers{9}.transferFcn = 'tansig';
% netOpr.layers{10}.transferFcn = 'tansig';

netOpr.layers{end}.transFerFcn = 'purelin';

netOpr.trainFcn = 'trainlm';

netOpr.trainParam.epochs = 50; % número máximo de épocas de treinamento
%% Divida os dados em conjuntos de treinamento, validação e teste:

%netOpr.divideFcn = ''; % divisão aleatória dos dados
%netOpr.divideMode = 'sample'; % divisão por amostra
netOpr.divideParam.trainRatio =  0.7;
netOpr.divideParam.valRatio =  0.15;
netOpr.divideParam.testRatio = 0.15;

%% Numeros:
%% Treine a rede neural: 

%net.trainParam.goal = 0.01; % meta de erro de treinamento
%net.trainParam.showWindow = true; % mostrar a janela de treinamento
[netNum, trNb] = train(netNum, inNumbers, trNumbers); % treinamento da rede neural

out = sim(netNum, inNumbers);
plotconfusion(trNumbers,out);

plotperf(trNb)         % Grafico com o desempenho da rede nos 3 conjuntos           

r=0;
for i=1:size(out,2)               % Para cada classificacao  
  [a, b] = max(out(:,i));          %b guarda a linha onde encontrou valor mais alto da saida obtida
  [c, d] = max(trNumbers(:,i));  %d guarda a linha onde encontrou valor mais alto da saida desejada
  if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
  end
end

accuracy_total_num = r/size(out,2)*100;
fprintf('Precisao total (numbers): %f\n', accuracy_total_num);

% Teste a rede neural:
TInputNumbers = inNumbers(:, trNb.testInd);
TTargetsNumbers = trNumbers(:, trNb.testInd);

out = sim(netNum, TInputNumbers);

%erro = perform(net, out,TTargets);
%fprintf('Erro na classificação do conjunto de teste %f\n', erro)

%Calcula e mostra a percentagem de classificacoes corretas no conjunto de teste
r=0;
for i=1:size(trNb.testInd,2)               % Para cada classificacao  
  [a, b] = max(out(:,i));          %b guarda a linha onde encontrou valor mais alto da saida obtida
  [c, d] = max(TTargetsNumbers(:,i));  %d guarda a linha onde encontrou valor mais alto da saida desejada
  if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
  end
end
accuracy_teste_num = r/size(trNb.testInd,2)*100;
fprintf('Precisao teste (numbers) %f\n', accuracy_teste_num);

str1 = 'train_nr_two_nn_numbers';
str2 = num2str(iteracao);
result = strcat(str1, '', str2);
save(result, 'netNum');

%% Operators 

%net.trainParam.goal = 0.01; % meta de erro de treinamento
%net.trainParam.showWindow = true; % mostrar a janela de treinamento
[netOpr, trOpr] = train(netOpr, inOperators, trOperators); % treinamento da rede neural


out = sim(netOpr, inOperators);
plotconfusion(trOperators,out);

plotperf(trOpr)         % Grafico com o desempenho da rede nos 3 conjuntos           

r=0;
for i=1:size(out,2)               % Para cada classificacao  
  [a, b] = max(out(:,i));          %b guarda a linha onde encontrou valor mais alto da saida obtida
  [c, d] = max(trOperators(:,i));  %d guarda a linha onde encontrou valor mais alto da saida desejada
  if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
  end
end

accuracy_total_opr = r/size(out,2)*100;
fprintf('\nPrecisao total (operators): %f\n', accuracy_total_opr);

% Teste a rede neural:
TInputOperators = inOperators(:, trOpr.testInd);
TTargetsOperators = trOperators(:, trOpr.testInd);

out = sim(netOpr, TInputOperators);

%erro = perform(net, out,TTargets);
%fprintf('Erro na classificação do conjunto de teste %f\n', erro)

%Calcula e mostra a percentagem de classificacoes corretas no conjunto de teste
r=0;
for i=1:size(trOpr.testInd,2)                  % Para cada classificacao  
  [a, b] = max(out(:,i));                      %b guarda a linha onde encontrou valor mais alto da saida obtida
  [c, d] = max(TTargetsOperators(:,i));        %d guarda a linha onde encontrou valor mais alto da saida desejada
  if b == d                                    % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
  end
end
accuracy_teste_opr = r/size(trOpr.testInd,2)*100;
fprintf('Precisao teste (operators) %f\n', accuracy_teste_opr);

str1 = 'train_nr_two_nn_operators';
str2 = num2str(iteracao);
result = strcat(str1, '', str2);
save(result, 'netOpr');

avg_accuracy_total = (accuracy_total_num + accuracy_total_opr) / 2;
avg_accuracy_teste = (accuracy_teste_num + accuracy_teste_opr) / 2;
fprintf('\nmedia da precisao total: %f', avg_accuracy_total);
fprintf('\nmedia da precisao teste: %f', avg_accuracy_teste);

end