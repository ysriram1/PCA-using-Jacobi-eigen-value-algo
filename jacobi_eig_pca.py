from math import sin, cos, atan, pi
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
        # calculate angle of rotation (theta)
        try:
            theta = math.atan(2*M_updated[i,j]/(M_updated[j,j]-M_updated[i,i]))/2
        except: # incase the denominator is 0
            if M_updated[i,j] >= 0:
                theta = math.pi/4
            else:
                theta = -math.pi/4
        # create corresponding Givens' Matrix
        Q = create_givens_mat(n,i,j,theta)
        # update Q (this will be the eigen vectors)
        Q_updated = np.dot(Q_updated, Q)
        # get udpated M using Q.T*M*Q
        M_updated = np.dot(np.dot(Q.T, M_updated),Q)

    # get indices of most to least significant eig vals for sorting
    sort_indices = M_updated.diagonal().argsort()[::-1]

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
        X = X - np.matlib.repmat(np.mean(X,0),n,1)
    if scale:
        X = np.divide(X,np.matlib.repmat(np.std(X,0),n,1))
    # find the covariance Matrix
    covM = 1./n * np.dot(X.T, X)
    # find the eigen values and vectors
    eigen_vals, pcs = jacobi_eig(covM)
    # only return the top k pcs
    return np.dot(X, pcs[:k].T)


########################## TESTING ###########################################
# test jacobi_eig()
import numpy as np
# 1
A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] # symmetric psd
# using numpy
vals, vecs = np.linalg.eig(A)
print(vals)
#[ 3.41421356  2.          0.58578644]
print(vecs.T)
#[[ -5.00000000e-01   7.07106781e-01  -5.00000000e-01]
# [ -7.07106781e-01   4.05405432e-16   7.07106781e-01]
# [  5.00000000e-01   7.07106781e-01   5.00000000e-01]]
# using above func
vals, vecs = jacobi_eig(Matrix(A))
print(vals)
#[3.414213562373095, 1.9999999999999996, 0.5857864376269049]
print(vecs)
#[[0.5000000000000526, 0.7071067811864732, 0.5000000000000526], 
# [0.7071067811865474, -4.629821043762799e-17, -0.7071067811865476], 
# [0.4999999999999474, -0.7071067811866218, 0.4999999999999475]]

# 2
B = [[2, 1, 0], [1, 4, 1], [0, 1, 4]] # symmetric psd
# using numpy
vals, vecs = np.linalg.eig(B)
print(vals)
#[ 1.5188057   3.31110782  5.17008649]
print(vecs.T)
#[[ 0.88765034 -0.42713229  0.17214786]
# [ 0.39711255  0.52065737 -0.75578934]
# [ 0.23319198  0.73923874  0.63178128]]
# using above func
vals, vecs = jacobi_eig(Matrix(B))
print(vals)
#[5.170086486626034, 3.311107817465982, 1.5188056959079843]
print(vecs)
#[[0.8876503388301478, -0.42713228703499645, 0.17214785896716067], 
# [0.39711254978700716, 0.5206573684395941, -0.7557893406837775], 
# [0.2331919783705817, 0.7392387395569922, 0.6317812811106418]]


# test pca()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
iris_data = np.loadtxt('iris_data.csv',delimiter=',',usecols=range(4))
# using sklearn
scaled_iris = StandardScaler().fit_transform(iris_data) # center and scale
res = PCA(n_components=2).fit_transform(scaled_iris)
print(res)
#[[-2.26454173  0.5057039 ]
# [-2.0864255  -0.65540473]
# [-2.36795045 -0.31847731]
# ..., 
# [ 1.52084506  0.26679457]
# [ 1.37639119  1.01636193]
# [ 0.95929858 -0.02228394]]
# using above function
X = Matrix(iris_data.tolist())
res = pca(X, k=2, center=True, scale=True)
print(res.tolist()[:3]) # displaying only first 3
#[[-2.2645417284068214, 0.5057039035404838], 
# [-2.086425500669378, -0.6554047279403764], 
# [-2.3679504490768926, -0.31847731117093403]]

###############################################################################
