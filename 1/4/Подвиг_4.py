import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)  # псевдослучайные числа образуют одну и ту же последовательность (при каждом запуске)
x = np.arange(-1.0, 1.0, 0.1)  # аргумент [-1; 1] с шагом 0,1

size_train = len(x)  # размер выборки
w = [0.5, -0.3]  # коэффициенты

model_a = lambda m_x, m_w: (m_w[1] * m_x + m_w[0])  # модель -0.3 * x + 0.5
loss = lambda ax, y: (ax - y) ** 2  # квадратическая функция потерь

y = model_a(x, w) + np.random.normal(0, 0.1, size_train)  # целевые значения
y_pred = model_a(x, w)

Q = loss(y_pred, y).mean()

sns.lineplot(x=x, y=y_pred, color='red')
sns.scatterplot(x=x, y=y)
plt.show()
