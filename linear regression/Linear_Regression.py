from time import time
start = time()
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

fig1 = plt.figure()
ax1 = Axes3D(fig1)
ax1.scatter(age, sex, num, color='#004467')
ax1.set_xlabel("Age",labelpad=5)
ax1.set_ylabel("Sex",labelpad=5)
ax1.set_zlabel("Disease",labelpad=5)
ax1.set_title("Age vs. Sex vs. Disease",loc='center',pad=5)
fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.scatter(cp, trestbps, num, color='#004467')
ax2.set_xlabel("Cp",labelpad=5)
ax2.set_ylabel("Trestbps",labelpad=5)
ax2.set_zlabel("Disease",labelpad=5)
ax2.set_title("Cp vs. Trestbps vs. Disease",loc='center',pad=5)
fig3 = plt.figure()
ax3 = Axes3D(fig3)
ax3.scatter(chol, fbs, num, color='#004467')
ax3.set_xlabel("Cholestrol",labelpad=5)
ax3.set_ylabel("Fbs",labelpad=5)
ax3.set_zlabel("Disease",labelpad=5)
ax3.set_title("Cholestrol vs. Fbs vs. Disease",loc='center',pad=5)
fig4 = plt.figure()
ax4 = Axes3D(fig4)
ax4.scatter(restecg, thalach, num, color='#004467')
ax4.set_xlabel("RestECG",labelpad=5)
ax4.set_ylabel("Thalach",labelpad=5)
ax4.set_zlabel("Disease",labelpad=5)
ax4.set_title("RestECG vs. Thalach vs. Disease",loc='center',pad=5)
fig5 = plt.figure()
ax5 = Axes3D(fig5)
ax5.scatter(exang, oldpeak, num, color='#004467')
ax5.set_xlabel("Exang",labelpad=5)
ax5.set_ylabel("Oldpeak",labelpad=5)
ax5.set_zlabel("Disease",labelpad=5)
ax5.set_title("Exang vs. Oldpeak vs. Disease",loc='center',pad=5)
fig6 = plt.figure()
ax6 = Axes3D(fig6)
ax6.scatter(slope, thal, num, color='#004467')
ax6.set_xlabel("Slope",labelpad=5)
ax6.set_ylabel("Thal",labelpad=5)
ax6.set_zlabel("Disease",labelpad=5)
ax6.set_title("Slope vs. Thal vs. Disease",loc='center',pad=5)
#plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, thal]).T
Y = np.array(num)

reg = LinearRegression()

reg = reg.fit(X, Y)

Y_pred = reg.predict(X)

rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)
r2 *= 100
print("Root Measn Squared Error => ",rmse)
print("Accuracy => ",r2)
stop = time()
print("Time => ",(stop-start)*1000,"Microsecond")