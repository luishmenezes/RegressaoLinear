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


lr_full = LinearRegression()
lr_full.fit(X, y)
coef_no_scaling = lr_full.coef_


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
pipe.fit(X, y)
coef_scaling = pipe.named_steps['lr'].coef_

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coef sem padronização": coef_no_scaling,
    "Coef com padronização": coef_scaling,
    "Abs sem padronização": np.abs(coef_no_scaling),
    "Abs com padronização": np.abs(coef_scaling)
})

print("\n Exercício 2 ")
print(coef_df.sort_values("Abs com padronização", ascending=False))