#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:46:01 2019

@author: javier
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .utils import check_symmetric, check_is_fitted
def gower(D):
    """
    Compute Gower matrix
    
    """
    
    # Dimensionality of distance matrix
    n = int(D.shape[0])

    # Create Gower's symmetry matrix (Gower, 1966)
    A = -0.5*np.square(D)

    # Subtract column means As = (I - 1/n * 11")A
    As = A - np.outer(np.ones(n),  A.mean(0))
    #Substract row
    G = As - np.outer(As.mean(1), np.ones(n))

    return G

def hat_matrix(X):
    """
    Calculates distance-based hat matrix for an NxM matrix of M predictors from
    N variables. Adds the intercept term for you.
    
    """
    N = X.shape[0]
    
    X = np.column_stack((np.ones(N), X)) # add intercept
    
    XXT = np.matmul(X.T, X)
    XXT_inv = np.linalg.inv(XXT)
    
    H = np.matmul(np.matmul(X, XXT_inv), X.T)

    return H

def compute_SSB(H, G, df1):
    """
    Compute between group sum of squares
    
    """
    
    trace_HG = np.trace(np.matmul(H, G))
    return trace_HG/df1
    
def compute_SSW(H, G, df2):
    """
    Compute within group sum of squares
    
    """
    trace_HG = np.trace(np.matmul(H, G))
    trace_G  = np.trace(G)

    return (trace_G - trace_HG )/df2

def design_matrices(df):
    """
    Construct design matrix from dataframe
    
    """
    
    X_list = []
    for ii in  range(df.shape[1]):
        #TODO: category gives error
        if (df.iloc[:, ii].dtype=='object'):
            X_list.append(pd.get_dummies(df.iloc[:, ii], 
                                         drop_first=True).values)
        else:
            X_list.append(df.iloc[:, ii].values[:, np.newaxis])
            
    return X_list

class MDMR(object):
    
    def __init__(self, 
                 n_perms = None, 
                 random_state=None, 
                 n_jobs = None, 
                 verbose=0):
        
        self.n_perms = n_perms
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def fit(self, X_df, D):
         
        # Check input is a dataframe
        if isinstance(X_df, pd.DataFrame) is False:
            raise AttributeError("Input data must be a dataframe")

        # Check if D is symmetrical and convert to numpy
        D = np.asarray(D)
        check_symmetric(D)
            
        # Compute Gower matrix and its trace for later
        G = gower(D)
        trG = np.trace(G)
        
        # save variables' names
        self.vars_ = np.array(X_df.columns)
        
        # Extract list of design matrices from the dataframe. This is done
        # to be able to handle categorical features
        X_list = design_matrices(X_df)
    
        # Full model hat matrix
        X_full = np.column_stack(X_list)
        
        # N observations and m features (without intercept)
        N, m = X_full.shape
        
        H_full = hat_matrix(X_full)
        df2 = N - m
        
        print("Computing omnibus statistic...")
        den = compute_SSW(H_full, G, df2) 
        
        # Compute SSB for full model (omnibus)
        num_omni = compute_SSB(H_full, G, m)
        
        # Compute F and R2 for omnibus model
        self.F_omni_ = num_omni/den
        self.r2_omni_ = num_omni*m/trG
        self.df_omni_ = m
        
        print("Computing individual variable statistic...")
        # Compute differences between H and defreees of freedom
        H_list = []
        for ii in range(len(X_list)):
            temp = X_list.copy()
            temp.pop(ii)
            if temp:
                H_ii = H_full - hat_matrix(np.column_stack(temp))
            else:
                H_ii = H_full 
            H_list.append(H_ii)
            
        # Compute degrees of freedom
        df1_list = []
        for X in X_list:
            m_ii = X.shape[1]
            df1_list.append(m_ii)
        
        self.df_ = np.array(df1_list)
        
        
        # Compute SSB for each column
        num_x = Parallel(n_jobs=self.n_jobs, 
                         verbose=self.verbose)(delayed(compute_SSB)(H, G, df1) for \
                                           (H, df1) in zip(H_list, df1_list))            
        num_x = np.array(num_x)
            
        self.F_ = num_x/den
        # pseudo R2.Note that we have to multiply by the degrees of freedom
        self.r2_ = np.multiply(num_x, np.array(df1_list))/trG
        
        if self.n_perms:
            # Generate indices
            np.random.seed(self.random_state)
            idxs_perm = [np.random.choice(a=N, size=N, replace=False) \
                             for ii in range(self.n_perms)]
            
            print("Generating null model by reshuffling the distance matrix")
            G_perm = Parallel(n_jobs=self.n_jobs,
                              verbose=self.verbose)(delayed(gower)\
                                                  (D[idxs,:][:,idxs]) \
                                                      for idxs in idxs_perm) 
                                                    
            print("Computing p-value for onmibus effect...")
            den_perm = Parallel(n_jobs=self.n_jobs,
                                verbose=self.verbose)(delayed(compute_SSW)\
                                                    (H_full, G, df2) \
                                                        for G in G_perm)
            den_perm = np.array(den_perm)
            num_omni_perm = Parallel(n_jobs=self.n_jobs, 
                                     verbose=self.verbose)(delayed(compute_SSB)\
                                                    (H_full, G, m) \
                                                        for G in G_perm)
            num_omni_perm = np.array(num_omni_perm)
            
            self.F_perm_omni_ = num_omni_perm/den_perm
            
            print("computing p-value for each variable...")
            num_x_perm = np.zeros(shape=(self.n_perms, len(H_list)))
            for ii, (H, df1) in enumerate(zip(H_list, df1_list)):    
                num_x_perm[:,ii] = Parallel(n_jobs=self.n_jobs, 
                                            verbose=self.verbose)(delayed(compute_SSB)\
                                                                (H, G, df1)\
                                                                for G in G_perm)
            
            self.F_perm_ = num_x_perm/den_perm[:, np.newaxis]
            
            pval_omni_ = np.sum(self.F_omni_ < self.F_perm_omni_)/self.n_perms
            
            pvals_ = [np.sum(self.F_[ii] < self.F_perm_[:,ii])/self.n_perms \
                for ii in range(len(self.F_))]
            
            self.pval_omni_ = pval_omni_
            self.pvals_ = np.array(pvals_, dtype=float)
        else:
            self.pval_omni_ = np.nan
            self.pvals_ = np.array([np.nan for ii in range(len(self.F_))])
                                                      
            
        return self
    
    def summary(self):
        
        check_is_fitted(self)
        
        summary_df = pd.DataFrame({'F': [self.F_omni_] + list(self.F_),
                                   'df': [self.df_omni_] + list(self.df_),
                                   'pseudo-R2': [self.r2_omni_] + list(self.r2_),
                                   'p-value':[self.pval_omni_] + list(self.pvals_)})
        
        summary_df.index = ['Omnibus'] + list(self.vars_)
        
        return print(summary_df)
