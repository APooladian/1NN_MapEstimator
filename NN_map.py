import ot
import numpy as np
from sklearn.neighbors import NearestNeighbors

#### 1-Nearest Neighbor Estimator
def LoopThruG(G,n,thr=1e-8):
  l1 = []
  l2 = []
  mx = G.max()
  for i in range(n):
    l1.append(i)
    for j in range(n):
      if G[i, j] / mx > thr:
        l2.append(j)
  return dict(zip(l1,l2))

def OT_0(source,target,a=None,b=None,maxiters=1000000):
    n = source.shape[0]
    if a == None:
        a = np.ones(n,)/n
    if b == None:
        b = np.ones(n,)/n
    M = ot.dist(source,target)
    G0 = ot.emd(a,b,M,numItermax=maxiters)
    return G0

def NNEstimator(x,source,target,G0,algo='brute'):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm=algo).fit(source)
    _, indices = nbrs.kneighbors(x)

    #target_1nn = np.zeros((len(indices),dim))
    target_1nn = np.zeros_like(x)
    loopG = LoopThruG(G0,len(G0))
    for i, e in enumerate(indices):
      ind_ = loopG[int(e)]
      target_1nn[i] = target[ind_]

    return target_1nn
