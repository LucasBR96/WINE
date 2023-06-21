import pandas as pd
import numpy as np

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