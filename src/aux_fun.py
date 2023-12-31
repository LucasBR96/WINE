import pandas as pd
import numpy as np
from typing import *

from itertools import product

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors as nn
from sklearn.base import clone

################### CLASSIFICATION #########################

def class_get_data( ):

    #---------------------------------------------------
    # Basic data
    data = pd.read_csv( "DATA/winequality.csv" )
    X = data.to_numpy()[ : , :-1 ]
    y = data.to_numpy()[ : , -1 ]

    #--------------------------------------------------
    # normalized data
    mu = X.mean( axis = 0 )
    sigma = X.std( axis = 0 )
    norm_X = ( X - mu )/sigma

    #-------------------------------------------------
    # pca form
    pca = PCA( 2 ).fit( norm_X )
    X_2d = pca.transform( norm_X )

    return X , y , X_2d , data.columns

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
    score = ( 3*precision + recall )/4
    return score , precision , recall


def knn_sythesis( X : np.ndarray , num_new : int , n_ngh = 5 ):
    m , n = X.shape

    # -----------------------------------
    # sampling minority class
    idxs = np.random.choice( m , num_new , replace = ( num_new > n ) )
    sample = X[ idxs ]

    #------------------------------------
    # getting the nearest neighbors of each
    # member of the sample
    knn : nn = nn( n_neighbors = n_ngh ).fit( X )
    neighs = knn.kneighbors( sample , return_distance = False ) 

    synth_X = np.zeros( ( num_new , n ) )
    for i in range( num_new ):

        #-----------------------------------
        # getting new element
        X_sample = X[ neighs[ i ] ]
        weights = np.random.random( n_ngh )
        weights = weights/weights.sum()
        synth_X[ i ] = weights@X_sample
    
    return synth_X

def smote_supersampling( X : np.ndarray , y : np.ndarray , n_ngh = 25 , seed = None , replace = True ):
    
    #-------------------------------------------
    # finding the minority class, binary case
    n = len( y )
    n_pos = y.sum( dtype = int )
    min_cls = 1 if n_pos <= n - n_pos else 0
    
    # --------------------------------------
    # Separating the minority class
    where = ( y == min_cls )
    X_min = X[ where ]
    n_min = n_pos if min_cls else n - n_pos

    #------------------------------------------
    # how many new samples of the minority class
    new = n - n_min
    if replace:
        new = 1 + new//2

    #-------------------------------------
    # getting synthetic data
    synth_X = knn_sythesis( X_min , new , n_ngh )

    #---------------------------------------
    # adding synthetic data
    new_X = X.copy()
    new_y = y.copy()

    if replace:
        maj_idx = np.arange( n )[ y != min_cls ]
        idxs = np.random.choice( maj_idx , new , replace = False )
        new_X[ idxs ] = synth_X
        new_y[ idxs ] = min_cls
    
    return new_X , new_y

def get_prepared_data( X : np.ndarray , y : np.ndarray , k = 10 , seed = None , smote = False):

    # making the k folder
    skf = StratifiedKFold( n_splits = k , random_state = seed , shuffle = True )
    skf_iter = skf.split( X , y )

    for _ in range( k ):

        # next fold
        train_idx , test_idx = next( skf_iter )
        X_train , y_train = X[ train_idx ] , y[ train_idx ]
        X_test , y_test = X[ test_idx ] , y[ test_idx ]

        # scaling X based on training set
        mu    = X_train.mean( axis = 0 )
        sigma = X_train.std( axis = 0 )
        X_train = ( X_train - mu )/sigma
        X_test = ( X_test - mu )/sigma

        # supersampling minority class with smote
        if smote:
            X_train , y_train = smote_supersampling( X_train , y_train )

        # returning pre processed data
        yield ( X_train , y_train ) , ( X_test , y_test )

# def results_to_csv( data : pd.DataFrame , path : str ):
    
#     arr_copy = np.copy(data.to_numpy())
#     for i in np.ndindex(arr_copy.shape):
#         arr_copy[i] = str(arr_copy[i]).replace('.', ',')

#     str_data = pd.DataFrame( data = arr_copy , columns = data.columns )
#     str_data.to_csv( path , sep = ";" )

def eval_model( folds : Iterator , model : object , eval_function : Callable  ):

    '''
    folds -> iterator for folded data set ( k-fold stratfied ), it is reco
    mended that the training data is small enought to be fitted at once

    model -> a sklearn ( or custom ) classification model that have already
    defined parameters but is not fitted yet

    eval_function -> returns a dict[ str , float ] that evaluates the performan
    ce of the model on several metrics
    '''

    # results for each fold
    lst = []
    for ( X_train , y_train ) , ( X_test , y_test ) in folds:

        # no object should fit twice
        m_clone = clone( model )
        m_clone.fit( X_train , y_train )

        #evaluating on test set
        y_hat = m_clone.predict( X_test )
        eval_metrics = eval_function( y_hat , y_test )
        lst.append( eval_metrics )
    
    return pd.DataFrame( lst )

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
