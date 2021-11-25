# https://angeloyeo.github.io/2019/07/27/PCA.html

import numpy as np
from numpy.matrixlib import defmatrix

# number = 5
# D = np.zeros(shape=(5,2))
# for i in range(5):
#     D[i] = input().split()
# print(D)

D = np.array([[170.,  70.],
            [150.,  45.],
            [160.,  55.],
            [180.,  60.],
            [170.,  80.]])


Dmean = D.T.mean(axis=1)
print(Dmean)
# Broadcasting
X = D-Dmean
print(X)
covar = np.dot(X.T, X)
print(covar)
covar = covar/D.shape[0]
print(covar)
