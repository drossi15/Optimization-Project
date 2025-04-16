% Importazione dati 
% Carica il file .mat
clc,clear
data = load('breast_cancer_data.mat');

% Accedo ai dati
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

% Rimuovo le colonne corrispondenti a w == 0 dalla matrice A
A= A(:, ~zero_weights);

% Grid Search

% Definisco gli intervalli per ciascun iperparametro
g_range = [1, 5, 10];        % Esempio per il parametro gamma del kernel sigmoidale
r_range = [0, 0.5, 1];       % Esempio per il parametro di spostamento
nu_range = [0.1, 0.5, 1];      % Esempio per il parametro di regolarizzazione

% Inizializzo variabili per salvare il miglior risultato
bestAccuracy = -Inf;
bestParams = struct('g', NaN, 'r', NaN, 'nu', NaN);

cv = cvpartition(length(d), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx  = test(cv);


% Grid search
for g = g_range
    for r = r_range
        for nu = nu_range

            A_train = A(trainIdx, :);
            d_train = d(trainIdx);
            A_test  = A(testIdx, :);
            d_test  = d(testIdx);
            
            % Definisco la matrice diagonale con le etichette di training
            D_train = diag(d_train);
            
            % Calcola le matrici kernel per il training e il test con il kernel sigmoidale
            K_train = tanh(g * (A_train * A_train') + r);
            K_test  = tanh(g * (A_test * A_train') + r);
            
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
            
            % Calcola le predizioni sul test set
            y_pred = sign(K_test * D_train * u - gam);
            accuracy = sum(y_pred == d_test) / length(d_test);
            
            % Visualizza il risultato per questa combinazione di iperparametri
            fprintf('g: %.2f, r: %.2f, nu: %.2f -> Accuracy: %.4f\n', g, r, nu, accuracy);
            
            % Aggiorna i migliori parametri se l'accuracy è migliore
            if accuracy > bestAccuracy
                bestAccuracy = accuracy;
                bestParams.g = g;
                bestParams.r = r;
                bestParams.nu = nu;  
            

            end
        end
    end
end

% Visualizza i migliori iperparametri trovati
fprintf('Migliori parametri: g=%.2f, r=%.2f, nu=%.2f con accuracy=%.4f\n', bestParams.g, bestParams.r, bestParams.nu, bestAccuracy);
