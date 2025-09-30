import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

alphas = [0.1, 1.0, 10.0]
results = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha, max_iter=5000)

    ridge_scores = cross_val_score(ridge, X, y, cv=5, scoring="r2")
    lasso_scores = cross_val_score(lasso, X, y, cv=5, scoring="r2")

    results.append({
        "Modelo": f"Ridge(alpha={alpha})",
        "R² médio": ridge_scores.mean()
    })
    results.append({
        "Modelo": f"Lasso(alpha={alpha})",
        "R² médio": lasso_scores.mean()
    })

results_df = pd.DataFrame(results)

print("\n=== Exercício 3 — Regularização ===")
print(results_df)
