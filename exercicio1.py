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

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Exercício 1")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

