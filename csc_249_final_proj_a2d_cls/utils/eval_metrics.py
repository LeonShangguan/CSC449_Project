import numpy as np

# data format
# The dimension of X_pre and X_gt are NXnum_cls, whether N is the number of samples and num_cls is the number of classes

def Precision(X_pre, X_gt):

    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x*y)/(np.sum(x) + 1e-8)
    return p/N

def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / (np.sum(y)+ 1e-8)
    return p/N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2*np.sum(x * y) / (np.sum(x) + np.sum(y)+ 1e-8)
    return p/N
