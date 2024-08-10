import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x_test = [(5, -3), (-3, 8), (3, 6), (0, 0), (5, 3), (-3, -1), (-3, 3)]

X = np.column_stack((np.ones(len(x_test)), x_test))
w = np.array([-33, 9, 13])
predict = np.sign(X @ w)

line = [33 / 13 - 9 / 13 * x for x in range(-4, 6)]

plt.plot(list(range(-4, 6)), line, color='red')
sns.scatterplot(x=[x[0] for x in x_test], y=[x[1] for x in x_test], hue=predict, palette='muted')
plt.show()