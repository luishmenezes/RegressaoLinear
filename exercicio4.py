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

X_subset = X[['MedInc', 'HouseAge', 'AveRooms']]

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y, test_size=0.2, random_state=0
)

lr_plot = LinearRegression()
lr_plot.fit(X_train, y_train)
y_pred_plot = lr_plot.predict(X_test)

residuos = y_test - y_pred_plot

plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
plt.scatter(y_pred_plot, residuos, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valores preditos")
plt.ylabel("Resíduos")
plt.title("Resíduos vs Preditos")


plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_plot, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.xlabel("Valores reais")
plt.ylabel("Valores preditos")
plt.title("Real vs Predito")

plt.tight_layout()
plt.show()