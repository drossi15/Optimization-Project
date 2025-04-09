%% Importazione dati 
% Carico il file .mat
clc,clear
data = load('breast_cancer_data.mat');

% Accedo ai dati
A = data.X;
d = data.y;
d = cellfun(@(x) 1 * strcmp(x, 'M') - 1 * strcmp(x, 'B'), d);
A=normalize(A);

%% Feature Selection

[m, n] = size(A);  % dimensioni dei dati
nu=0.1;
D = diag(d);



cvx_begin quiet
    cvx_solver Mosek   % utilizzo di Mosek come risolutore
    variables w(n) gam s(n) y(m)
    
    % Funzione obiettivo: minimizzazione di nu*sum(y) + sum(s)
    minimize( nu*sum(y) + sum(s) )
    
    subject to
        D * (A*w - gam*ones(m,1)) + y >= ones(m,1);
        -s <= w <= s;
        y >= 0;
cvx_end


zero_weights = (w == 0);

% Rimuovo le colonne corrispondenti a w == 0 dalla matrice A
A= A(:, ~zero_weights);

%% Gaussian Kernel SVM with 10 fold Cross Validation



sigma = 0.01; % Iperparametro del kernel
K = exp(-sigma * squareform(pdist(A, 'euclidean').^2)); % Matrice kernel
nu=1;

% Imposto la cross-validation con K = 10
cv = cvpartition(length(d), 'KFold', 10);

%  variabili per raccogliere i risultati 
accuracies = zeros(cv.NumTestSets, 1);
precisions   = zeros(cv.NumTestSets, 1);
recalls      = zeros(cv.NumTestSets, 1);
f1_scores    = zeros(cv.NumTestSets, 1);
train_accuracies=zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    % indici per il training e il test
    trainIdx = cv.training(i);
    testIdx = cv.test(i);

    % Dati di training e di test
    A_train = A(trainIdx, :);
    d_train = d(trainIdx);
    A_test = A(testIdx, :);
    d_test = d(testIdx);
    D_train=diag(d_train);
    D_test=diag(d_test);

    l_train=length(d_train);

    %la matrice kernel gaussiano
    K_train = exp(-sigma * pdist2(A_train, A_train, 'euclidean').^2);
    K_test = exp(-sigma * pdist2(A_test, A_train, 'euclidean').^2);
    
    cvx_begin quiet
    cvx_solver mosek
    variables u(l_train) gam y(l_train) s(l_train)

    % Funzione obiettivo con norma L1 e kernel
    minimize(nu * sum(y) + sum(s))

    % Vincoli con il kernel
    subject to
        D_train * (K_train * D_train * u - gam*ones(l_train,1)) + y >= ones(l_train,1);
        -s <= u <= s; % Vincolo per la norma L1
        y >= 0;
    cvx_end
    
   
    y_pred = sign(K_test * D_train * u - gam);
    y_pred_train = sign(K_train * D_train * u - gam);

    accuracies(i) = sum(y_pred == d_test) / length(d_test);
    train_accuracies(i)=sum(y_pred_train == d_train) / length(d_train);

    % Calcolo i componenti per precision, recall e F1 score
    %  "positivo" la classe +1
    TP = sum((d_test == 1) & (y_pred == 1));
    FP = sum((d_test == -1) & (y_pred == 1));
    TN = sum((d_test == -1) & (y_pred == -1));
    FN = sum((d_test == 1) & (y_pred == -1));
    
    % Evito la divisione per zero
    if (TP + FP) == 0
        precision = 0;
    else
        precision = TP / (TP + FP);
    end
    if (TP + FN) == 0
        recall = 0;
    else
        recall = TP / (TP + FN);
    end
    if (precision + recall) == 0
        f1 = 0;
    else
        f1 = 2 * precision * recall / (precision + recall);
    end
    
    % Salvo i risultati per questa fold
    precisions(i) = precision;
    recalls(i)    = recall;
    f1_scores(i)  = f1;
end

% Calcolo i valori medi su tutte le fold
mean_accuracy_gaussian = mean(accuracies);
mean_precision = mean(precisions);
mean_recall = mean(recalls);
mean_f1 = mean(f1_scores);
mean_train_accuracy = mean(train_accuracies);

% Visualizza i risultati
disp(['Accuratezza media della cross-validation: ', num2str(mean_accuracy_gaussian)]);
disp(['Precision media della cross-validation: ', num2str(mean_precision)]);
disp(['Recall media della cross-validation: ', num2str(mean_recall)]);
disp(['F1 Score medio della cross-validation: ', num2str(mean_f1)]);
disp(['Accuratezza di train medio della cross-validation: ', num2str(mean_train_accuracy)]);

% Dati delle metriche
metriche = [mean_train_accuracy*100, mean_accuracy_gaussian*100, mean_precision*100, mean_recall*100, mean_f1*100];

% Nomi delle metriche
nomi_metriche = {'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score'};

% Creazione del grafico a barre
figure;
bar(metriche);
set(gca, 'XTickLabel', nomi_metriche);
ylabel('Accuracy');
title('Confronto delle Metriche della SVM Kernel Gaussiano');
ylim([50 100]);
grid on;

for i = 1:length(metriche)
    text(i, metriche(i) + 1, sprintf('%.2f', metriche(i)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end



%% Polynomial Kernel SVM with 10 fold Cross Validation

% Parametri per il kernel polinomiale
c = 0.1;       % Costante
p = 3;       % Grado del polinomio
nu=0.01;

% Imposta la cross-validation con K = 10
cv = cvpartition(length(d), 'KFold', 10);

% Inizializzo vettori per raccogliere i risultati
accuracies = zeros(cv.NumTestSets, 1);
precisions   = zeros(cv.NumTestSets, 1);
recalls      = zeros(cv.NumTestSets, 1);
f1_scores    = zeros(cv.NumTestSets, 1);
train_accuracies=zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    % indici per il training e il test
    trainIdx = cv.training(i);
    testIdx = cv.test(i);

    % Dati di training e di test
    A_train = A(trainIdx, :);
    d_train = d(trainIdx);
    A_test = A(testIdx, :);
    d_test = d(testIdx);
    
    D_train = diag(d_train);
    
    % matrice kernel polinomiale per il training
    K_train = (A_train * A_train' + c) .^ p; % Dimensione: m_train x m_train
    
    % matrice kernel polinomiale per il test
    K_test = (A_test * A_train' + c) .^ p;    % Dimensione: m_test x m_train
    
    % Numero di campioni nel training
    m_train = length(d_train);
    
    
    cvx_begin quiet
        cvx_solver mosek
        variables u(m_train) gam y(m_train) s(m_train)
        
        % Funzione obiettivo: minimizzazione di 0.1*sum(y) + sum(s)
        minimize(nu * sum(y) + sum(s))
        
        % Vincoli con il kernel polinomiale:
        subject to
            D_train * (K_train * D_train * u - gam * ones(m_train,1)) + y >= ones(m_train,1);
            -s <= u <= s;  % Vincolo per la norma L1
            y >= 0;
    cvx_end
    
    % Predizione sul test set:
    y_pred = sign(K_test * D_train * u - gam);
    y_pred_train = sign(K_train * D_train * u - gam);
    
    % Calcolo l'accuratezza per questa fold
    accuracies(i) = sum(y_pred == d_test) / length(d_test);
    train_accuracies(i)=sum(y_pred_train == d_train) / length(d_train);
    
    % Calcolo i componenti per precision, recall e F1 score
    % "positivo" la classe +1
    TP = sum((d_test == 1) & (y_pred == 1));
    FP = sum((d_test == -1) & (y_pred == 1));
    TN = sum((d_test == -1) & (y_pred == -1));
    FN = sum((d_test == 1) & (y_pred == -1));
    
    % Evito la divisione per zero
    if (TP + FP) == 0
        precision = 0;
    else
        precision = TP / (TP + FP);
    end
    if (TP + FN) == 0
        recall = 0;
    else
        recall = TP / (TP + FN);
    end
    if (precision + recall) == 0
        f1 = 0;
    else
        f1 = 2 * precision * recall / (precision + recall);
    end
    
    % Salvo i risultati per questa fold
    precisions(i) = precision;
    recalls(i)    = recall;
    f1_scores(i)  = f1;
end

% Calcolo i valori medi su tutte le fold
mean_accuracy_polynomial = mean(accuracies);
mean_precision = mean(precisions);
mean_recall = mean(recalls);
mean_f1 = mean(f1_scores);
mean_train_accuracy = mean(train_accuracies);

% Visualizza i risultati
disp(['Accuratezza di test media della cross-validation: ', num2str(mean_accuracy_polynomial)]);
disp(['Precision media della cross-validation: ', num2str(mean_precision)]);
disp(['Recall media della cross-validation: ', num2str(mean_recall)]);
disp(['F1 Score medio della cross-validation: ', num2str(mean_f1)]);
disp(['Accuratezza di train medio della cross-validation: ', num2str(mean_train_accuracy)]);

% Dati delle metriche
metriche = [mean_train_accuracy*100, mean_accuracy_polynomial*100, mean_precision*100, mean_recall*100, mean_f1*100];

% Nomi delle metriche
nomi_metriche = {'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score'};

% Creazione del grafico a barre
figure;
bar(metriche);
set(gca, 'XTickLabel', nomi_metriche);
ylabel('Accuracy');
title('Confronto delle Metriche della SVM Kernel Polinomiale');
ylim([50 100]); % Imposta il range da 0 a 1
grid on;

for i = 1:length(metriche)
    text(i, metriche(i) + 1, sprintf('%.2f', metriche(i)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end

%% Sigmoid Kernel SVM with 10 fold Cross Validation

% Parametri per il kernel sigmoid
g = 5;   % Parametro gamma (regola l'influenza dei singoli punti)
r = 1;          % Costante di spostamento
nu=0.1;

% Imposto la cross-validation con K = 10
cv = cvpartition(length(d), 'KFold', 10);

% Inizializzo vettori per raccogliere i risultati
accuracies = zeros(cv.NumTestSets, 1);
precisions   = zeros(cv.NumTestSets, 1);
recalls      = zeros(cv.NumTestSets, 1);
f1_scores    = zeros(cv.NumTestSets, 1);
train_accuracies = zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    %indici per il training e il test
    trainIdx = cv.training(i);
    testIdx = cv.test(i);

    % Dati di training e di test
    A_train = A(trainIdx, :);
    d_train = d(trainIdx);
    A_test = A(testIdx, :);
    d_test = d(testIdx);
    
    
    D_train = diag(d_train);
    
    %  matrice kernel sigmoide per il training
    K_train = tanh(g * (A_train * A_train') + r); 
    
    %  matrice kernel sigmoide per il test
    K_test = tanh(g * (A_test * A_train') + r);    
    
    % Numero di campioni nel training
    m_train = length(d_train);
    
    
    cvx_begin quiet
        cvx_precision high
        cvx_solver mosek
        variables u(m_train) gam y(m_train) s(m_train)
        
        % Funzione obiettivo: minimizzazione di 0.1*sum(y) + sum(s)
        minimize(nu * sum(y) + sum(s))
        
        % Vincoli con il kernel sigmoide:
        subject to
            D_train * (K_train * D_train * u - gam * ones(m_train,1)) + y >= ones(m_train,1);
            -s <= u <= s;  % Vincolo per la norma L1
            y >= 0;
    cvx_end
    
    % Predizione sul test set:
    y_pred = sign(K_test * D_train * u - gam);
    y_pred_train = sign(K_train * D_train * u - gam);
    
    % Calcolo l'accuratezza per questa fold
    accuracies(i) = sum(y_pred == d_test) / length(d_test);
    train_accuracies(i) = sum(y_pred_train == d_train) / length(d_train);
    
    % Calcolo i componenti per precision, recall e F1 score
    % Definiamo "positivo" come la classe +1
    TP = sum((d_test == 1) & (y_pred == 1));
    FP = sum((d_test == -1) & (y_pred == 1));
    TN = sum((d_test == -1) & (y_pred == -1));
    FN = sum((d_test == 1) & (y_pred == -1));
    
    % Evita la divisione per zero
    if (TP + FP) == 0
        precision = 0;
    else
        precision = TP / (TP + FP);
    end
    if (TP + FN) == 0
        recall = 0;
    else
        recall = TP / (TP + FN);
    end
    if (precision + recall) == 0
        f1 = 0;
    else
        f1 = 2 * precision * recall / (precision + recall);
    end
    
    % Salvo i risultati per questa fold
    precisions(i) = precision;
    recalls(i)    = recall;
    f1_scores(i)  = f1;
end

% Calcolo i valori medi su tutte le fold
mean_accuracy_sigmoid = mean(accuracies);
mean_precision = mean(precisions);
mean_recall = mean(recalls);
mean_f1 = mean(f1_scores);
mean_train_accuracy = mean(train_accuracies);

% Visualizza i risultati
disp(['Accuratezza di test media della cross-validation: ', num2str(mean_accuracy_sigmoid)]);
disp(['Precision media della cross-validation: ', num2str(mean_precision)]);
disp(['Recall media della cross-validation: ', num2str(mean_recall)]);
disp(['F1 Score medio della cross-validation: ', num2str(mean_f1)]);
disp(['Accuratezza di train medio della cross-validation: ', num2str(mean_train_accuracy)]);

% Dati delle metriche
metriche = [mean_train_accuracy*100, mean_accuracy_sigmoid*100, mean_precision*100, mean_recall*100, mean_f1*100];

% Nomi delle metriche
nomi_metriche = {'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score'};

% Creazione del grafico a barre
figure;
bar(metriche);
set(gca, 'XTickLabel', nomi_metriche);
ylabel('Accuracy');
title('Confronto delle Metriche della SVM Kernel Sigmoidale');
ylim([50 100]); % Imposta il range da 0 a 1
grid on;

for i = 1:length(metriche)
    text(i, metriche(i) + 1, sprintf('%.2f', metriche(i)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end


%%


% Dati delle metriche
metriche = [mean_accuracy_gaussian*100, mean_accuracy_polynomial*100, mean_accuracy_sigmoid*100];

% Nomi delle metriche
nomi_metriche = {'Test Accuracy_ Gaussian', 'Test Accuracy_ Polynomial', 'Test Accuracy_ Sigmoid'};

% Creazione del grafico a barre
figure;
bar(metriche);
set(gca, 'XTickLabel', nomi_metriche);
ylabel('Accuracy');
title('Confronto fra Test Accuracy');
ylim([50 100]); % Imposta il range da 0 a 1
grid on;

for i = 1:length(metriche)
    text(i, metriche(i) + 1, sprintf('%.2f', metriche(i)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end