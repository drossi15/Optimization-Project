% Importazione dati 
% Carica il file .mat
clc,clear
data = load('breast_cancer_data.mat');

% Accedi ai dati
A = data.X;
d = data.y;
d = cellfun(@(x) 1 * strcmp(x, 'M') - 1 * strcmp(x, 'B'), d);
A=normalize(A);

% Feature Selection

[m, n] = size(A);  % dimensioni dei dati
nu=0.1;
D = diag(d);


cvx_begin
    cvx_solver Mosek   % specifica l'utilizzo di Mosek come risolutore
    variables w(n) gam s(n) y(m)
    
    % Funzione obiettivo: minimizzazione di nu*sum(y) + sum(s)
    minimize( nu*sum(y) + sum(s) )
    
    subject to
        D * (A*w - gam*ones(m,1)) + y >= ones(m,1);
        -s <= w <= s;
        y >= 0;
cvx_end


zero_weights = (w == 0);

% Rimuovi le colonne corrispondenti a w == 0 dalla matrice A
A= A(:, ~zero_weights);

% Grid Search

% Definizione dei range degli iperparametri
sigma_range = [0.01, 0.05, 0.1, 0.5, 1]; % Possibili valori di sigma
nu_range = [0.01, 0.1, 0.5, 1]; % Possibili valori di nu

% Inizializza variabili per il miglior modello
bestAccuracy = -Inf;
bestParams = struct('sigma', NaN, 'nu', NaN);

% Suddivisione Train-Test (80% - 20%)
cv = cvpartition(length(d), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

A_train = A(trainIdx, :);
d_train = d(trainIdx);
A_test = A(testIdx, :);
d_test = d(testIdx);
D_train = diag(d_train);

% Loop sui parametri per trovare la miglior combinazione
for sigma = sigma_range
    for nu = nu_range
        
        % Calcola la matrice kernel Gaussiana
        K_train = exp(-sigma * pdist2(A_train, A_train, 'euclidean').^2);
        K_test = exp(-sigma * pdist2(A_test, A_train, 'euclidean').^2);
        
        l_train = length(d_train);

        % Risolvi il problema duale con il kernel Gaussiano
        cvx_begin quiet
            cvx_solver mosek
            variables u(l_train) gam y(l_train) s(l_train)
            minimize(nu * sum(y) + sum(s))
            subject to
                D_train * (K_train * D_train * u - gam*ones(l_train,1)) + y >= ones(l_train,1);
                -s <= u <= s;
                y >= 0;
        cvx_end

        % Predizione sul test set
        y_pred = sign(K_test * D_train * u - gam);
        
        % Calcola l'accuratezza
        accuracy = sum(y_pred == d_test) / length(d_test);
        fprintf('sigma: %.2f, nu: %.2f -> Accuracy: %.4f\n', sigma, nu, accuracy);

        % Se è il miglior modello, lo salviamo
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestParams.sigma = sigma;
            bestParams.nu = nu;
        end
    end
end

% Stampa i migliori iperparametri trovati
disp(['Miglior sigma: ', num2str(bestParams.sigma)]);
disp(['Miglior nu: ', num2str(bestParams.nu)]);
disp(['Miglior accuratezza: ', num2str(bestAccuracy)]);
