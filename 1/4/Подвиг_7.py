import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)  # псевдослучайные числа образуют одну и ту же последовательность (при каждом запуске)
x = np.arange(-1.0, 1.0, 0.1)  # аргумент [-1; 1] с шагом 0,1

model_a = lambda xx, ww: (ww[0] + ww[1] * xx + ww[2] * xx ** 2 + ww[3] * xx ** 3)  # модель
Y = np.sin(x * 5) + 2 * x + np.random.normal(0, 0.1, len(x))  # вектор целевых значений

X = np.array([[1, xx, xx**2, xx**3] for xx in x])  # обучающая выборка для поиска коэффициентов w модели a
w = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

y_pred = model_a(x, w)

plt.scatter(x=x, y=Y, color='blue')
plt.plot(x, y_pred, color='red')

poly = PolynomialFeatures(4, include_bias=True)  # возможно degree = 5? Или переобучение?
x_poly = poly.fit_transform(x.reshape(-1, 1), Y)

model = LinearRegression()
model.fit(x_poly, Y)

sklearn_y_pred = model.predict(x_poly)
print(y_pred, sklearn_y_pred)
plt.plot(x, sklearn_y_pred)
plt.show()