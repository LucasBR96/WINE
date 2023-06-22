import pandas as pd
import numpy as np
from typing import *

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

def bin_f1_score( guess : np.ndarray , classes : np.ndarray ):

    #--------------------------------------
    # basic metrics
    tp : int = ( guess*classes ).sum()
    fp : int = ( guess*( 1 - classes ) ).sum()
    fn : int = ( ( 1 - guess )*classes ).sum()

    #---------------------------------------
    # terms
    precision = tp/( tp + fp )
    recall    = tp/( tp + fn )
    return 2*precision*recall/( precision + recall )

def get_prepared_data( df : pd.DataFrame , target : pd.Series , k = 10 , seed = None ):

    # df contains de discriminatory data
    X = df.to_numpy()

    # target is the class data
    y = target.to_numpy()

    # making the k folder
    skf = StratifiedKFold( n_splits = k , random_state = seed )
    skf_iter = skf.split( X , y )

    for _ in range( k ):

        # next fold
        train_idx , test_idx = next( skf_iter )
        X_train , y_train = X[ train_idx ] , y[ train_idx ]
        X_test , y_test = X[ test_idx ] , y[ test_idx ]

        # scaling X based on training set
        X_max = X_train.max( axis = 0 )
        X_min = X_train.min( axis = 0 )
        X_train = ( X_train - X_min )/( X_max - X_min )
        X_test = ( X_test - X_min )/( X_max - X_min )

        # returning pre processed data
        yield ( X_train , y_train ) , ( X_test , y_test )


################### CLUSTERIZATION #########################

def cluster_get_data( ):

    #---------------------------------------------------
    # Basic data
    data = pd.read_csv( "DATA/winequality.csv" )
    X = data.to_numpy()[ : , :-1 ]

    #--------------------------------------------------
    # normalized data
    mu = X.mean( axis = 0 )
    sigma = X.std( axis = 0 )
    norm_X = ( X - mu )/sigma

    #-------------------------------------------------
    # pca form
    pca = PCA( 2 ).fit( norm_X )
    X_2d = pca.transform( norm_X ).T

    return X , norm_X , X_2d , data.columns[ :-1 ]

def cluster_stats( X , columns ):

    df = pd.DataFrame( index = columns[ 1: ] , columns = [ "mean" , "std" ] )
    XT = X.T
    for i , att in enumerate( columns ):

        mean = XT[ i ].mean()
        if att == "is_red":
            red_probs = mean
            continue
        
        std = XT[ i ].std()
        df.loc[ att ] = pd.Series( [ f"{ mean:.5}" , f"{ std:.3f}" ] , index = [ "mean" , "std"] )
    return df , red_probs

def cluster_score( X : np.ndarray ):

    '''
    The mean distance between the points of X and its 
    centroid
    '''

    #centroid
    mu = X.mean( axis = 0 )
    
    # distances
    d_sqr = ( X - mu )**2
    d = np.sqrt( d_sqr.sum( axis = 1 ) )

    return d.mean()

# TODO extra for if the project is done ahead of schedule 
# def mean_si_coef( X : np.ndarray , labels : np.ndarray ):

#     '''
#     only works for binary cluesters
#     '''

#     # coeficcient for every cluster
#     n = int( labels.max() )
#     silouette = np.zeros( n )

#     for cls in range( n ):

#         where = ( labels == cls )
        
#         # selecting members of the cluster
#         X_cls = X[ where ]

#         # a[ i , j , k ] = X_cls[ i , k ] - X_cls[ j , k ]
#         a = np.expand_dims( X_cls , 1 ) - X_cls 
#         a_sqr = ( a )