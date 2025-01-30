'''
Report (in LaTeX): https://www.overleaf.com/read/tsqfzbkcmqkv#cb9a84
To execute this file, keep the dataset file, main.py and functions.py in the same directory and use 
python3 main.py command in the terminal. 
'''

# importing libraries
import numpy as np 
import pandas as pd 
from scipy import sparse
import matplotlib.pyplot as plt

# The file functions.py has helper functions to preprocess dataset and to to matrix multiplications 
from functions import *

'''
Data Acquisition 
From the drawing made by hand - a X-Y 2D dataset is been generated using: https://wpd.starrydata2.org/
The dataframe contains two columns 'X' and 'Y' in the range of [1,1000]
'''
# Data Loading and Cleaning (i.e.) Preprocessing 
df  = pd.read_csv('dataset.csv') 
df = preprocess_data(df)

print('Shape of the pandas dataframe\t', df.shape, '\n')
print('Column Names:\t', df.columns)

# Changing the column names in the dataset to make plotting easier 
df.rename(
    columns = {'77.011149825784':'X',' 414.9792834365889':'Y'},
    inplace = True
)
print('Changed Column Names:\t', df.columns, '\n')

# Visualisation - Plotting the X-Y dataset to see the image as a scatter plot 
df.plot.scatter(
    x = 'X', 
    y = 'Y',   
)
plt.savefig('input_image_scatter.png')
print('Scatter plot of the input image saved!\n')

# Transformation - Sparse Martrix cnstruction and matrix multiplication 
sparse_matrix = sparse.csr_matrix(df)
print('Shape of the Sparse Matrix:\t', sparse_matrix.shape,'\n')

# Rotating the matrix and saving the file in the same directory.  

# Clockwise
rotated_image = rotate_matrix(sparse_matrix, 'clockwise')
rotated_image = pd.DataFrame(rotated_image, columns=['X', 'Y'])
rotated_image.plot.scatter(
    x = 'X', 
    y = 'Y',   
)
plt.savefig('clockwise_rotated_image_scatter.png')
print('Scatter plot of the rotated image saved!\n')

# Counter - clockwise
rotated_image = rotate_matrix(sparse_matrix, 'counter-clockwise')
rotated_image = pd.DataFrame(rotated_image, columns=['X', 'Y'])
rotated_image.plot.scatter(
    x = 'X', 
    y = 'Y',   
)
plt.savefig('counter_clockwise_rotated_image_scatter.png')
print('Scatter plot of the rotated image saved!\n')

# Flip
rotated_image = rotate_matrix(sparse_matrix, 'flip')
rotated_image = pd.DataFrame(rotated_image, columns=['X', 'Y'])
rotated_image.plot.scatter(
    x = 'X', 
    y = 'Y',   
)
plt.savefig('flip_rotated_image_scatter.png')
print('Scatter plot of the rotated image saved!\n')

print('')