import warnings
warnings.filterwarnings("ignore")

import platform 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from lazypredict.Supervised import LazyRegressor


from functions import *
print('\n#-------------------------- START OF ASSIGNMENT 3 --------------------------#\n')
print('\nThe plots for this assignment are saved in the same directory, when the code is executed they are shown one after another.\n')

if __name__ == '__main__':

    # Loading Dataset 
    df = pd.read_csv('Assignment3.csv')
    print('Dataset is loaded!\n')

    # Data cleaning 
    missing_values_table(df)

    #-------------------------- Task 1 --------------------------#
    
    # Creating Arrays to do regression 
    print('\n1. Applying OLS Regression on the given dataset and calculating loss:\n')

    X1 = np.array(df[['x1', 'x2', 'x3', 'x4', 'x5']])
    Y = np.expand_dims(df['y'], 1)

    print('\n\tShape of X:\t', X1.shape)
    print('\tShape of Y:\t', Y.shape)

    print('\n\tOLS Model, without bias\n')
    OLS_model = LinearRegression(fit_intercept=False).fit(X1, Y)

    m_OLS = OLS_model.coef_
    print('\tOLS Solution of weights:\t', m_OLS)
    y_pred_1 = OLS_model.predict(X1)
    m_OLS_RMSE = Loss.RMSE(Y, y_pred_1)
    m_OLS_SSE = Loss.SSE(Y, y_pred_1)
    print('\n\tRMSE:\t', m_OLS_RMSE)
    print('\tSSE:\t', m_OLS_SSE)
    
    print('\n\tOLS Model, with bias\n')
    OLS_model = LinearRegression(fit_intercept=True).fit(X1, Y)

    m_OLS = OLS_model.coef_
    print('\tOLS Solution of weights:\t', m_OLS)
    y_pred_1 = OLS_model.predict(X1)
    m_OLS_SSE = Loss.SSE(Y, y_pred_1)
    m_OLS_RMSE = Loss.RMSE(Y, y_pred_1)
    print('\n\tRMSE:\t', m_OLS_RMSE)
    print('\tSSE:\t', m_OLS_SSE)

    print(
        '\n',
        '\t* When OLS is done with the given dataset with five features x1, x2, x3, x4, x5 and label y, RMSE is approx 26. The SSE is Approx 71k.'
    )

    #-------------------------- Task 2 --------------------------#

    print('\n2. Performing EDA (Exploratory Data Analysis):\n')

    print('\t* To understnad the relationship / dependence between the features, the concepts and covariance and correlation are used.')
    print('\t* From the correlation matrix, we understand that feature x1, x2 and x4 are linearly realted to the label y\n')

    # Using the heatmap function in seaborn library to plot the covairance matrix
    sns.heatmap(df.cov(), cmap="YlGnBu", annot=True)
    plt.title('Covariance Matrix - calculated for both features & label')
    plt.savefig('Covariance_Matrix_With_labels.png')
    plt.show() 
    plt.close()

    # Using the heatmap function in seaborn library to plot the correlation matrix
    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
    plt.title('Correlation Matrix - calculated for both features & label')
    plt.savefig('Correlation_Matrix_With_labels.png')
    plt.show() 
    plt.close()

    # Dropping the labels from the dataset 
    df_features = pd.DataFrame({
        'x1': df.x1, 
        'x2': df.x2, 
        'x3': df.x3, 
        'x4': df.x4, 
        'x5': df.x5
    })
    
    # Using the heatmap function in seaborn library to plot the covairance matrix
    sns.heatmap(df_features.cov(), cmap="YlGnBu", annot=True)
    plt.title('Covariance Matrix - calculated only with features.')
    plt.savefig('Covariance_Matrix_Without_labels.png')
    plt.show() 
    plt.close()

    # Using the heatmap function in seaborn library to plot the correlation matrix
    sns.heatmap(df_features.corr(), cmap="YlGnBu", annot=True)
    plt.title('Correlation Matrix - calculated only with features')
    plt.savefig('Correlation_Matrix_Without_labels.png')
    plt.show() 
    plt.close()

    # Using pairplot function from seaborn to check the relationships between variables 
    sns.pairplot(df_features)
    plt.title('Pair plot - between features')
    plt.savefig('PairPlot_Between_Features.png')
    plt.close()

    print(
        '\n \t* From the correlationa and pair plot, we understand that the feature x3 has zero correltion and thus it can be dropped.', 
        '\n \t* Features x2 and x5 are realted by a quadratic relation. As we can observe from the pair plot, x5 is a parabola centered at origin in terms of x2'
    )
    print(
        '\t* Thus the feature x3 is dropped and we model another feature as x2^2 or sqrt(x5)' 
    )

    df_modified_features = pd.DataFrame({
        '$x_1$'        : df.x1, 
        '$x_2$'        : df.x2, 
        '$x_2^{2}$'    : np.square(df.x2), 
        '$x_4$'        : df.x4, 
        '$x_5$'        : df.x5,
        '$\sqrt{x_5}$' : np.sqrt(df.x5)
    })  

    sns.heatmap(df_modified_features.corr(), cmap="YlGnBu", annot=True)
    plt.title('Correlation between fearure transformations.png')
    plt.savefig('CorreltionMatrix_Transformed_Features.png')
    plt.close()

    print(
        '\t* After doing feature transformations, from the correlation matrix, we understand that features x1, x2, x2^2, x4 and x5 have high correlation.\n', 
        '\t* Thus, we use these as features to find the OLS Soltuion.\n' 
    )

    #-------------------------- Task 3 --------------------------#
    print('\n3. OLS Regression for transformed data:\n')

    print('Modified features, max of degree 2\n')
    X2 = np.array(df_modified_features[['$x_1$', '$x_2$', '$x_2^{2}$', '$x_4$', '$x_5$']]) 
    OLS_model = LinearRegression(fit_intercept=True).fit(X2, Y)

    m_OLS = OLS_model.coef_
    print('\tOLS Solution of weights:\t', m_OLS)
    y_pred_2 = OLS_model.predict(X2)
    m_OLS_RMSE = Loss.RMSE(Y, y_pred_2)
    m_OLS_SSE = Loss.SSE(Y, y_pred_2)
    print('\n\tRMSE:\t', m_OLS_RMSE)
    print('\tSSE:\t', m_OLS_SSE)

    print('Polynomial features, max of degree 6\n')
    poly_transformer = PolynomialFeatures(degree = 6)

    X21 = np.array(df[['x1','x2','x4','x5']])
    X21 = poly_transformer.fit_transform(X21)
    model = LinearRegression(fit_intercept=True).fit(X21, Y)
    y_pred = model.predict(X21)
    print('\tOLS Solution of weights:\t', model.coef_)
    m_Poly_RMSE = Loss.RMSE(Y, y_pred)
    m_Poly_SSE = Loss.SSE(Y, y_pred)
    print('\n\tRMSE:\t', m_Poly_RMSE)
    print('\tSSE:\t', m_Poly_SSE)

    print(
        '\n \tThe comaparing the baseline SSE and the SSE of polynomial features, we observe that there is significant difference in loss. The SSE of the regressor with polynomial features is less. It is probably overfitting at this point.\n'
    )

    #-------------------------- Task 4 --------------------------#
    # Lazypredict module mainly works on 32 bit environment, to prevent errors due to install libmop

    if(platform.architecture()[0] != '64bit'): 
        print('Install libomp and try again!')

    print('\nCheck Lazy_Regressor_1.csv and Lazy_Regressor_2.csv file(s) for the output of the Lazt Regressor Results!\n')
    
    # splittind the given dataset into test and train data to use in LazyRegressor
    # Mannual feature transformations 
    train_X, test_X, train_y, test_y = train_test_split(X2, Y,  test_size = 0.8, random_state = 42)
    
    lazy_model  = LazyRegressor(
        verbose = -1, 
        ignore_warnings = True, 
        custom_metric = None,
    )

    models, predictions = lazy_model.fit(train_X, test_X, train_y, test_y)

    predictions.to_csv('Lazy_Regressor_1.csv')
    
    # Using Polynomial Features 
    train_X, test_X, train_y, test_y = train_test_split(X21, Y,  test_size = 0.8, random_state = 42)
    
    lazy_model  = LazyRegressor(
        verbose = -1, 
        ignore_warnings = True, 
        custom_metric = None,
    )

    models, predictions = lazy_model.fit(train_X, test_X, train_y, test_y)

    predictions.to_csv('Lazy_Regressor_1.csv')
    print('\n\n\tResults from Lazy Regressor are saved!\n')

    print('\tInferences from Results:\n')
    print(
        '\n* Some features in the given dataset are correlated linearly (x1, x2, x4) and some were correlated quadratically (x2, x5)', 
        '\n* As we model the label as a linear combination of features, the SSE in this regression as high as 70k',
        '\n* Once the features are mapped to higher dimensions, let\'s say in two dimensions, there is significant change in SSE.',
        '\n* As we progress over more and more dimensions, the mapping becomes very close the actual realtionship and the SSE goes down near to zero.', 
        '\n* In the case of LazyPredict, the model takes all the values as a whole and traines them in batches over all types of available regressors in Sklearn without doing feature transformations.', 
        '\n* Since there are no feature transformations, the loss from models in Lazy predict is same as OLS without feature transformations!', 
        '\n* Once we apply transformed data to lazy predict, the loss decreases.', 
        '\n* It is inferred that, features which doesn\'t have any explicit realtionships in lower dimesnions, their relationship is exposed in higher dimensions.\n'
    )

print('\n#-------------------------- END OF ASSIGNMENT 3 --------------------------#\n')