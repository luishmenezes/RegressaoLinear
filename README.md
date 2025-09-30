Com base no dataset California Housing

Faça as seguintes atividades:
Exercício 1 — Baseline com subset de features
Tarefa: Use apenas as features ['MedInc', 'HouseAge', 'AveRooms'] para treinar um modelo LinearRegression.

Faça o split treino/teste (20% teste, random_state=0).
Calcule MAE, RMSE e R² no conjunto de teste.

Exercício 2 — Padronização e impacto nos coeficientes
Tarefa: Crie um Pipeline(StandardScaler() -> LinearRegression) e ajuste com todas as features.

Compare os coeficientes (em valor absoluto) com o modelo sem padronização.
Comente o que muda na interpretação.

Exercício 3 — Regularização
Tarefa: Com todas as features, avalie Ridge e Lasso com alpha ∈ {0.1, 1.0, 10.0} usando validação cruzada (CV=5).

Reporte a média do R² para cada combinação.

Exercício 4 - Plotting
Tarefa: Plotar resíduos e real vs predito.
