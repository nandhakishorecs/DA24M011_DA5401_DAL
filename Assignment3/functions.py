import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# looking for NaN / missing values in dataset 
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'}
    )
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values.")
    

class Loss: 
    def RMSE(y: np.ndarray, y_pred: np.ndarray) -> float: 
        error = 0 
        for i in range(0, len(y_pred)): 
            error += (y[i] - y_pred[i])**2
        error = np.sqrt(float(error/len(y_pred)))
        return error 

    def SSE(y: np.ndarray, y_pred: np.ndarray) -> float: 
        error = 0 
        for i in range(0, len(y_pred)): 
            error += (y[i] - y_pred[i])**2
        return error 