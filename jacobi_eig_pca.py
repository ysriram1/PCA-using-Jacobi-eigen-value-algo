from math import sin, cos, atan
import numpy as np

# https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
def jacobi_eig(M, ap=1e-6, rp=1e-5):
    '''
    returns the eigen values and their corresponding eigen vectors using
    the Jacobi algorithm. Each row is an eigen vector.
    Keyword Arguments:
    -- M: a symmetric, postive semi-definite matrix. Numpy Array
    -- ap: the absolute precision
    -- rp: the relative precision
    '''
    n,m = M.shape
    assert n == m, 'Matrix is not square'
    assert np.sum(M == M.T) != (m+1)*(n+1), 'Matrix is not symmetric'

    # returns an nxn Givens' rotation matrix based on plane (i,j)
    # https://en.wikipedia.org/wiki/Givens_rotation
    def create_givens_mat(n,i,j, theta):
        Q = np.zeros([n,n])
        # fill in the diagnols
        for k in range(n):
            if k not in [i,j]:
                Q[k,k] = 1
        Q[i,i] = cos(theta)
        Q[j,j] = cos(theta)
        Q[i,j] = sin(theta)
        Q[j,i] = -sin(theta)
        return Q

    M_updated = M
    Q_updated = np.identity(n)
    while True:
        # find max off-diagnol value in M
        max_val = np.abs(np.triu(M_updated,k=1)).max()
        # get its index
        max_val_idx = np.abs(np.triu(M_updated,k=1)).argmax()
        i,j = np.unravel_index(max_val_idx, M_updated.shape)
        # stopping condition
        if max_val < ap or max_val < rp*max_val:
            break
        # calculate theta
        theta = atan(M_updated[i,j]/(M_updated[j,j]-M_updated[i,i]))
        # create corresponding Givens' Matrix
        Q = create_givens_mat(n,i,j,theta)
        # update Q (this will be the eigen vectors)
        Q_updated = np.dot(Q_updated, Q)
        # get udpated M using Q.T*M*Q
        M_updated = np.dot(np.dot(Q.T, M_updated),Q)

    # get indices of most to least significant eig vals for sorting
    sort_indices = M_updated.diagonal().argsort()

    return M_updated.diagonal()[sort_indices], Q_updated.T[sort_indices]

def pca(X, k, center=True, scale=True):
    '''
    returns the first k scores of the Matrix X using PCA
    Keyword Arguments:
    -- X: the data matrix (array) where each row is a datapoint and each column is a feature
    -- k: number (int) of principal components to use to reduce the dimention of the data.
            If rank of Cov(X) < k, then the maxmimum possible PCs are used
    -- center: a boolean indicating whether to center the data to 0
    -- scale: a boolean indicating whether to scale the data
    '''
    n,m = X.shape
    if center:
        X = X - np.matlib.repmat(np.mean(X,1),n,1)
    if scale:
        X = np.divide(X,np.matlib.repmat(np.std(X,1),n,1))
    # find the covariance Matrix
    CovM = 1./n * np.dot(X.T, X)
    # find the eigen values and vectors
    pcs, eigen_vals = jacobi_eig(CovM)
    # only return the top k pcs
    return np.dot(X, pcs[:k].T)
