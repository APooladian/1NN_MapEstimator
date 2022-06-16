import numpy as np
from scipy.linalg import inv, sqrtm
from numpy.random import multivariate_normal as mvn

### sample from the uniform distribution between [amin,amin+range]
def sample_uniform(n,d,amin=-1,range=2):
    return np.random.rand(n,d)*range + amin

### sample from the normal distribution with covariance A and mean 0 
def sample_mvn(n,A):
    dim=A.shape[0]
    return mvn(mean=np.zeros(dim,), cov=A, size=(n,))

### coordinate-wise exponential map
def OT_exp(x):
    return np.exp(x)

### transport map between N(0,A) and N(0,B):
### to use on a vector of samples x:
### C = G2G_map(A,B)
### map = (C @ x.T).T returns target points in dimension n x d

def G2G_map(A,B):
    Asqrt = sqrtm(A)
    Asqrtinv = inv(Asqrt)
    return Asqrtinv @ sqrtm(Asqrt @ B @ Asqrt) @ Asqrtinv

    


