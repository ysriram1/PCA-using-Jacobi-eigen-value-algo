from math import sin, cos, atan
import numpy as np

# https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
def jacobi_eig(M, ap=1e-6, rp=1e-5):
    '''
    returns the eigen values and eigen vectors using the Jacobi algorithm.
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
        # get udpated M using Q.T*M*Q
        M_updated = np.dot(np.dot(Q.T, M_updated),Q)

    return M_updated.diagonal()
