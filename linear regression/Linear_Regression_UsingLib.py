import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('dataset.csv')
print(data.shape)
data.head()

age = data['age'].values
sex = data['sex'].values
cp = data['cp'].values
trestbps = data['trestbps'].values
chol = data['chol'].values
fbs = data['fbs'].values
restecg = data['restecg'].values
thalach = data['thalach'].values
exang = data['exang'].values
oldpeak = data['oldpeak'].values
slope = data['slope'].values
thal = data['thal'].values
num = data['num'].values

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(age, sex, trestbps, color='#ef1234')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, thal]).T
Y = np.array(num)

reg = LinearRegression()

reg = reg.fit(X, Y)

Y_pred = reg.predict(X)


rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print("Root Measn Squared Error => ",rmse)
print("R2 Score => ",r2)