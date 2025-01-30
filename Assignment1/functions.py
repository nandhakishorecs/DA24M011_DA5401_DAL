import numpy as np 
import pandas as pd 
import scipy
import matplotlib.pyplot as plt

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame: 
    # checking for null values 
    if(df.isnull().values.any()): 
        print('\nThe dataset has missing / NaN values, do preprocesssing!\n')
    else: 
        print('\nThe dataset does not have any missing / NaN values, Proceed ahead!\n')
    return df


'''
Mathematical Intitution for the below function 
For rotation of the image, we multiply the image with a rotation matrix 
    R = [[ cosx, -sinx],
         [ sinx,  cosx]]
Clockwise , x = +90, Anti-Clockwise, x = -90; Flip x = 180
'''

def rotate_matrix(sparse_matrix: scipy.sparse._csr.csr_matrix, direction: str) -> scipy.sparse._csr.csr_matrix: 
    # using '@' operator to multiply sparse matrix and normal matrix. 
    rotated_matrix = []
    if(direction == 'clockwise'): 
        rotation_matrix = [[ 0, -1],
                           [ 1,  0]]
        rotated_matrix = sparse_matrix @ rotation_matrix
    elif (direction == 'counter-clockwise'):
        rotation_matrix = [[  0,  1],
                           [ -1,  0]]
        rotated_matrix = sparse_matrix @ rotation_matrix
    elif (direction == 'flip'): 
        rotation_matrix = [[ -1,  0],
                           [  0, -1]]
        rotated_matrix = sparse_matrix @ rotation_matrix
    return rotated_matrix
    
