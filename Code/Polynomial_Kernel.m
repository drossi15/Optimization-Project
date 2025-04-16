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


% Definizione dei range degli iperparametri
c_range = [0.1, 1, 10];    % Valori possibili per la costante c
p_range = [2, 3, 4, 5];    % Valori possibili per il grado del polinomio
nu_range = [0.01, 0.1, 0.5, 1]; % Valori per il parametro di regolarizzazione

% Inizializza variabili per il miglior modello
bestAccuracy = -Inf;
bestParams = struct('c', NaN, 'p', NaN, 'nu', NaN);

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
for c = c_range
    for p = p_range
        for nu = nu_range
            
            % Calcola la matrice kernel polinomiale
            K_train = (A_train * A_train' + c) .^ p;
            K_test = (A_test * A_train' + c) .^ p;

            m_train = length(d_train);

            % Risolvi il problema duale con il kernel polinomiale
            cvx_begin quiet
                cvx_solver mosek
                variables u(m_train) gam y(m_train) s(m_train)
                minimize(nu * sum(y) + sum(s))
                subject to
                    D_train * (K_train * D_train * u - gam* ones(m_train,1)) + y >= ones(m_train,1);
                    -s <= u <= s;
                    y >= 0;
            cvx_end

            % Predizione sul test set
            y_pred = sign(K_test * D_train * u - gam);
            
            % Calcola l'accuratezza
            accuracy = sum(y_pred == d_test) / length(d_test);
            fprintf('c: %.2f,p: %.2f, nu: %.2f -> Accuracy: %.4f\n',c,p, nu, accuracy);

            % Se è il miglior modello, lo salviamo
            if accuracy > bestAccuracy
                bestAccuracy = accuracy;
                bestParams.c = c;
                bestParams.p = p;
                bestParams.nu = nu;
            end
        end
    end
end

% Stampa i migliori iperparametri trovati
disp(['Miglior c: ', num2str(bestParams.c)]);
disp(['Miglior p: ', num2str(bestParams.p)]);
disp(['Miglior nu: ', num2str(bestParams.nu)]);
disp(['Miglior accuratezza: ', num2str(bestAccuracy)]);
