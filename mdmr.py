#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:46:01 2019

@author: javier
"""

import numpy as np
from joblib import Parallel, delayed

def gower(D): 
  # Convert distance object to matrix form
  #D = as.matrix(D)

  # Dimensionality of distance matrix
  n = int(D.shape[0])

  # Create Gower's symmetry matrix (Gower, 1966)
  A = -0.5*np.square(D)
  
  # Subtract column means As = (I - 1/n * 11")A
  As = A - np.outer(np.ones(n),  A.mean(0))
  #Substract row
  G = As - np.outer(As.mean(1), np.ones(n))
  
  return G



def gowers_matrix(D):
    """Calculates Gower's centered matrix from a distance matrix."""
    assert_square(D)

    n = float(D.shape[0])
    o = np.ones((int(n), 1))
    I = np.identity(int(n)) - (1/n)*o.dot(o.T)
    A = -0.5*(np.square(D))
    G = I.dot(A).dot(I)

    return(G)

def hat_matrix(X):
    """
    Caluclates distance-based hat matrix for an NxM matrix of M predictors from
    N variables. Adds the intercept term for you.
    """
    n = X.shape[0]
    
    
    X = np.column_stack((np.ones(n), X)) # add intercept
    
    XXT = np.matmul(X.T, X)
    XXT_inv = np.linalg.inv(XXT)
    
    H = np.matmul(np.matmul(X, XXT_inv), X.T)

    return H

def calc_F(H, G, m=None):
    """
    Calculate the F statistic when comparing two matricies.
    """
    #assert_square(H)
    #assert_square(G)

    n = H.shape[0]
    I = np.identity(n)
    IG = I-G

    if m:
        F = (np.trace(H.dot(G).dot(H)) / (m-1)) / (np.trace(IG.dot(G).dot(IG)) / (n-m))
    else:
        F = (np.trace(H.dot(G).dot(H))) / np.trace(IG.dot(G).dot(IG))

    return F

def mdmr(X, D, n_perm=100):
    H = hat_matrix(X)
    
    #omnibus effect 
    
    # Computational trick: H is idempotent, so H = HH. tr(ABC) = tr(CAB), so
    # tr(HGH) = tr(HHG) = tr(HG). Also, tr(AB) = vec(A)"vec(B), so
    trHG = np.matmul(H.flatten().T, G.flatten())

    # Numerical trick: tr((I-H)G(I-H)) = tr(G) - tr(HGH), so
    trG  = np.trace(G)
    
    F = trHG/(trG - trHG )
    
def _select(H, X, ii):
    
    Xs = X[:, ~ii]
    
class MDMR(object):
    
    def __init__(self, X, D, n_perms = 100, random_state=None, n_jobs = None):
        self.n_perms = n_perms
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def fit(self, X, D, G=None):
        
        m = X.shape[0]
        n = X.shape[1]
        
        H = hat_matrix(X)
    
        #omnibus effect 
        
        if G is None:
            G = gower(D)
    
        # Computational trick: H is idempotent, so H = HH. tr(ABC) = tr(CAB), so
        # tr(HGH) = tr(HHG) = tr(HG). Also, tr(AB) = vec(A)"vec(B), so
        trace_HG = np.matmul(H.flatten().T, G.flatten())

        # Numerical trick: tr((I-H)G(I-H)) = tr(G) - tr(HGH), so
        trace_G  = np.trace(G)
    
        F_omni = trace_HG/(trace_G - trace_HG )
        pseudo_r2_omni = trace_HG/trace_G 
        
    
      ##############################################################################
      ## COMPUTE CONDITIONAL TEST STATISTICS
      ##############################################################################
      # Get vectorized hat matrices for each conditional effect
      
      
      Hs = Parallel(n_jobs=self.n_jobs)(delayed(PON_NOMBRE)(H, X, ii) \ 
                    for ii in range(m)
      
      Hs <- parallel::mclapply(1:px, mc.cores = ncores, FUN = function(k) {
        x.rm <- which(test.inds == k)
        Xs <- X[,-x.rm]
        Hs <- .hat(Xs)
        c(H - Hs)
      })
    
      # Get DF of each test
      df <- unlist(parallel::mclapply(1:px, mc.cores = ncores, FUN = function(k) {
        x.rm <- which(test.inds == k)
        length(x.rm)
      }))
    
      # Compute SSD due to conditional effect
      numer.x <- unlist(
        parallel::mclapply(Hs, mc.cores = ncores, FUN = function(vhs) {
          crossprod(vhs, vg)
        }))
    
      # Rescale to get either test statistic or pseudo r-square
      f.x <- numer.x / denom
      pr2.x <- numer.x / trG
    
      # Combine test statistics and pseudo R-squares
      stat <- data.frame("stat" = c(f.omni, f.x),
                         row.names = c("(Omnibus)", unique.xnames))
      df <- data.frame("df" = c(p, df),
                       row.names = c("(Omnibus)", unique.xnames))
      pr2 <- data.frame("pseudo.Rsq" = c(pr2.omni, pr2.x),
                        row.names = c("(Omnibus)", unique.xnames))
            
        

def permute(H, G, n=10000):
    """
    Calculates a null F distribution from a symmetrically-permuted G (Gower's
    matrix), from the between subject connectivity distance matrix D, and a the
    H (hat matrix), from the original behavioural measure matrix X.
    The permutation test is accomplished by simultaneously permuting the rows
    and columns of G and recalculating F. We do not need to account for degrees
    of freedom when calculating F.
    """
    F_null = np.zeros(n)
    idx = np.arange(G.shape[0]) # generate our starting indicies

    for i in range(n):
        idx = np.random.permutation(idx)
        G_perm = reorder(G, idx, symm=True)
        F_null[i] = calc_F(H, G_perm)

    F_null.sort()

    return F_null

def variance_explained(H, G):
    """
    Calculates variance explained in the distance matrix by the M predictor
    variables in X.
    """
    assert_square(H)
    assert_square(G)

    variance = (np.trace(H.dot(G).dot(H))) / np.trace(G)

    return(variance)

def mdmr(X, Y):
    """
    Multvariate regression analysis of distance matricies: regresses variables
    of interest X (behavioural) onto a matrix representing the similarity of
    connectivity profiles Y.
    """
    if not full_rank(X):
        raise Exception('X is not full rank:\ndimensions = {}'.format(X.shape))

    X = standardize(X)   # mean center and Z-score all cognitive variables
    R = np.corrcoef(Y)   # correlations of Z-scored correlations, as in Finn et al. 2015.
    D = r_to_d(R)        # distance matrix of correlation matrix
    G = gowers_matrix(D) # centered distance matrix (connectivity similarities)
    H = hat_matrix(X)    # hat matrix of regressors (cognitive variables)
    F = calc_F(H, G)     # F test of relationship between regressors and distance matrix
    F_null = permute(H, G)
    v = variance_explained(H, G)

    return F, F_null, v

def backwards_selection(X, Y):
    """
    Performs backwards variable selection on the input data.
    """

    return False

def individual_importances(X, Y):
    """
    Runs MDMR individually for each variable. If the variable is deemed
    significant, the variance explained is recorded, otherwise it is reported
    as 0. Returns a vector of variance explained.
    """
    m = X.shape[1]
    V = np.zeros(m)
    for test in range(m):
        X_test = np.atleast_2d(X[:, test]).T # enforces a column vector
        F, F_null, v = mdmr(X_test, Y)
        thresholds = sig_cutoffs(F_null, two_sided=False)
        if F > thresholds[1]:
            V[test] = v
        else:
            V[test] = 0
        print('tested variable {}/{}'.format(test+1, m))

    return V