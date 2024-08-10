import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)  # псевдослучайные числа образуют одну и ту же последовательность (при каждом запуске)
x = np.arange(-1.0, 1.0, 0.1)  # аргумент [-1; 1] с шагом 0,1

model_a = lambda xx, ww: (ww[0] + ww[1] * xx)  # модель
Y = -5.2 + 0.7 * x + np.random.normal(0, 0.1, len(x))  # вектор целевых значений

X = np.column_stack((np.ones(len(x)), x))
w = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

y_pred = model_a(X, w)

sns.scatterplot(x=x, y=Y)
sns.lineplot(x=x, y=y_pred[:, 1], color='red')
plt.show()