import numpy as np

x_test = [(9, 6), (2, 4), (-3, -1), (3, -2), (-3, 6), (7, -3), (6, 2)]

B = (2, 0)
A = (7, 7)
w1 = B[1] - A[1]
w2 = -(B[0] - A[0])
w0 = -(w1 * A[0] + w2 * A[1])
w = np.array([w0, w1, w2])

X = np.column_stack((np.ones(len(x_test)), x_test))
predict = np.sign(X @ w)