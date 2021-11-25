import numpy as np

X = np.array([[0, 2, 2, 0, 0, 0, 0, 0, 0, 0], # I
             [2, 0, 0, 1, 0, 0, 0, 0, 0, 0], # ate
             [2, 0, 0, 0, 0, 0, 1, 0, 1, 0], # like
             [0, 1, 0, 0, 1, 1, 0, 0, 0, 0], # a
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 1], # banana
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 1], # chearry
             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # deep
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 1], # learning
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 1], # NLP
             [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]]) # .

U, S, V_t =  np.linalg.svd(X)

# print(U)
print(np.round(np.dot(U, np.transpose(U))))
# print(V_t)

k = 2
theta = sum(S[:k])/ sum(S)
print(f'theta : {theta}')

U = U[:, :k]

x = np.diag(S[:k])
US = np.dot(U, np.diag(S[:k]))
# print(V_t.shape)
# print(US.shape)
X_hat = np.dot(US, V_t[:k, :])
X_hat = np.round(X_hat)
# print(X_hat)
