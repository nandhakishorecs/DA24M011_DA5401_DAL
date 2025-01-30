#########################################################################################################################################################
print('\n#############################################################################################\n')
print('\nSTART OF ASSIGNMENT 2\n')
print('\nThe plots for this assignment are saved in the same directory, when the code is executed they are shown one after another.\n')

# Importing Libraries 
import pandas as pd # type: ignore
import numpy as np # type: ignore

from sklearn.preprocessing import PolynomialFeatures # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

import matplotlib.pyplot as plt  # type: ignore
from matplotlib import rc # type: ignore
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Code for OLS , SSE, and prediction writtern from scratch.
from LinearRegression_Self import *

# Code for grid search as a function written from scratch. 
from grid_search import * 

#########################################################################################################################################################
# Preprocessing Data

# Imporitng dataset 
df_raw = pd.read_csv('Assignment2_dataset.data', sep = '\t')
df_raw.describe()

# Adding the 'x' component (i.e.) Time component 
x = np.linspace(0, 26, 226)
df_raw = df_raw.assign(x = x)

# Recreating the plot from the question to check the scale of 'x' component 
plt.xlabel('Time')
plt.ylabel('System output')
plt.plot(x, df_raw['SpringPos'], '-')
plt.plot(x, df_raw['StockPrice'], '-')
plt.title('Given two cyber physical systems: Spring and Stock price ticker')
plt.legend(['SpringPos', 'StockPrice'], loc = 'lower right')
plt.grid()
plt.savefig('Two_Cyber_Physical_systems_plot.png')
plt.show()
plt.close()

# Removing the Stock price column, as we model the spring 
df_1 = df_raw.drop('StockPrice', axis = 1)
df_1.rename(
    columns = { 'SpringPos' : 'y' },
    inplace = True
)
df_1.columns

# Getting X and Y arrays
X1 = np.expand_dims(df_1.x, 1)
Y = np.array(df_1.y) 

#########################################################################################################################################################
print('\n#############################################################################################\n')
## TASK 1
print("\nTASK 1")
# 1. OLS SOLUTION

print('\n1. Regression using OLS - using matrix operations')
print('\tTaking time as linspace(0, 226, 26), no feature engineering done \n')

m_opt = Linear_Regression.OLS_Solution(X1, Y)
print('\tOLS Solution of weights:\t', m_opt)

y_pred_1 = Linear_Regression.predict(m_opt, Y)
m_SSE = Linear_Regression.SSE(Y, y_pred_1)
print('\tSum of Squares Error:\t', m_SSE, '\n')

# Plotting the OLS Fit
plt.plot(X1, Y, 'r-')
plt.plot(X1, y_pred_1, 'b-')
plt.xlabel('Time')
plt.ylabel('Spring Position')
plt.title('OLS solution for linear regression with X as time')
plt.legend(['Original Data', 'OLS Soltution'], loc = 'lower right')
plt.grid()
plt.savefig('OLS_fit.png')
plt.show()
plt.close()

# 2. GRID SEARCH (Angle vs SSE )
print('2. Grid Search\n')
grid_search_error, angle = grid_search(df=df_1, start_angle=0, end_angle=60, step_size=5)

print('\tOptimal ange is\t:', angle, 'degrees for the slope when SSE is\t', grid_search_error, 'at minimum\n')

# 3. SKLEARN LINEAR MODEL IMPLEMENTATION

print('3. Regression using Sklearn.linear_model.LinearRegression()\n')
print('\tRegression using Sklearn.linear_model.LinearRegression \n')
model_0 = LinearRegression(fit_intercept=False).fit(X1, Y)
y_pred_0 = model_0.predict(X1)

print('\tSklearn\'s Linear Regression Solution of weights:\t', model_0.coef_)
model_0SSE = Linear_Regression.SSE(Y, y_pred_0)
print('\tSum of Squares Error:\t', model_0SSE, '\n')

plt.plot(X1, Y, 'r-')
plt.plot(X1, y_pred_0, 'b-')
plt.xlabel('Time')
plt.ylabel('Spring Position')
plt.title('sklearn.linear_model.LinearRegression Fit')
plt.legend(['Original Data', 'Sklearn Solution'], loc = 'lower right')
plt.grid()
plt.savefig('sklearn_Fit.png')
plt.show()
plt.close()

# 4. COMPARISON OF ANSWERS FOR TASK 1 - subparts 1,2,3
print('4. Comparing weights from questions 1, 2 & 3:\n')
print(
    '\tAmong the OLS implementation (without feature engineering), Sklearn and the grid search for angle at m is miminum, the optimal value is the OLS fit.\n',
    '\tThe OLS implementation is giving the best value for slope among three methods as it directly computes the answer using closed form solution\n'
)

#########################################################################################################################################################
print('\n#############################################################################################\n')
# TASK 2
print('TASK 2')

# 1. SPLITTING DATA INTO TRAIN TEST AND VALIDATION FOR INTERPOLATION AND EXTRAPOLATION.

print(
    '\n1. The given dataset is split in a ratio of 70%, 20%, 10% as training, validation and test datasets.\n', 
    '\tThis is done using test_train_split module from sklearn.model_selection.'
)

# Refer line 

# 2. MODELLING THE CURVE AS A DAMPED SINE WAVE AND REGRESSSION USING OLS SOLUTION


print(
    '\n2. (a) The given data can be modeled as a damped wave function of the form:', r'e^{-bx} (cosax + sinax)', '\n',
    '\tTaking b = 0.05999 and a = 0.8661, we can fit the curve with minimal loss.\n'
)

# Modelling the curve as a damped spring equation 
bias_component = 10 * np.ones(226)
t = np.linspace(0, 26, 226)
trig_component = (np.sin(0.8661*t) + np.cos(0.8661*t))
exp_component = np.exp(-0.05999*t)
damped_wave_eq = exp_component * trig_component

df_20 = pd.DataFrame({
    "bias" : bias_component, 
    "x" : damped_wave_eq,
    "y" : df_raw['SpringPos']
})

X20 = np.array(df_20[["bias", "x"]]) 

m_wave_opt = Linear_Regression.OLS_Solution(X20, Y)
print('\tOLS Solution of weights:\t', m_wave_opt)
y_pred_20 = Linear_Regression.predict(m_wave_opt, X20)
m_wave_SSE = Linear_Regression.SSE(Y, y_pred_20)
print('\tSum of Squares Error:\t', m_wave_SSE, '\n')

plt.plot(X1, Y, 'r-')
plt.plot(X1, y_pred_20, 'b-')
plt.xlabel('Time')
plt.ylabel('Spring Position')
plt.title('OLS fit, where X is modelled as a damped spring equation')
plt.legend(['original', 'damped_spring_eq'], loc = 'lower right')
plt.grid()
plt.savefig('Improved_OLS_fit_damped_spring.png')
plt.show()
plt.close()



# Modeliing the original curve as a damped sine wave: 
bias_component = 10 * np.ones(226)
t = np.linspace(0, 26, 226)
trig_component = (np.sin(0.8661*t)) # + np.cos(0.8*t) 
exp_component = np.exp(-0.05999*t)
damped_wave_eq = exp_component * trig_component

print(
    '\n2. (b) The given data can be modeled as a damped sine function of the form:', r'e^{-bx} sinax', '\n',
    '\tTaking b = 0.05999 and a = 0.8661, we can fit the curve with minimal loss.\n'
)

df_2 = pd.DataFrame({
    "bias" : bias_component, 
    "x" : damped_wave_eq,
    "y" : df_raw['SpringPos']
})

X2 = np.array(df_2[["bias", "x"]]) 

m_sin_opt = Linear_Regression.OLS_Solution(X2, Y)
print('\tOLS Solution of weights:\t', m_sin_opt, )
y_pred_2 = Linear_Regression.predict(m_sin_opt, X2)
m_sin_SSE = Linear_Regression.SSE(Y, y_pred_2)
print('\tSum of Squares Error:\t', m_sin_SSE, '\n')

plt.plot(X1, Y, 'r-')
plt.plot(X1, y_pred_2, 'b-')
plt.xlabel('Time')
plt.ylabel('Spring Position')
plt.title('OLS fit, where X is modelled as a damped sine wave')
plt.legend(['original', 'damped_sin_eq'], loc = 'lower right')
plt.grid()
plt.savefig('Improved_OLS_fit.png')
plt.show()
plt.close()

print(
    '\n2. (c) The given data can also be modeled using polynomial features using the functions from sklearn.',
    '\n \tTaking taking degree as 25, we can fit the curve with the minimum loss.\n'
)


poly_transformer = PolynomialFeatures(degree = 25) # 25
X3 = np.expand_dims(df_2.x, 1)

X3 = poly_transformer.fit_transform(X3)
model = LinearRegression(fit_intercept=False).fit(X3, Y)
y_pred_3 = model.predict(X3)

model_SSE = Linear_Regression.SSE(Y, y_pred_3)
print('\tSum of Squares Error:\t', model_SSE)
print('\tOLS Solution of weights:\n\n', model.coef_, '\n')

plt.plot(X1, Y, 'r-')
plt.plot(X1, y_pred_3, 'b-')
plt.xlabel('Time')
plt.ylabel('Spring Position')
plt.title('OLS fit using polynomial function of degree 25.')
plt.legend(['original', 'model'], loc = 'lower right')
plt.grid()
plt.show()
plt.savefig('Polynomial_features_OLS_fit.png')
plt.close()

# INTERPOLATION 
# Shuffling is done and we predict the missing values in the regression line. 
print('Interpolation:\n')
train_x, temp_x, train_y, temp_y = train_test_split(X3, Y, test_size=0.2, random_state=42, shuffle=True)
test_x, val_x, test_y, val_y = train_test_split(temp_x, temp_y, test_size=0.7, random_state=42, shuffle=True)
model = LinearRegression(fit_intercept=False).fit(train_x, train_y)
prediction = model.predict(test_x)
print('\tLoss:\t', Linear_Regression.SSE(test_y, prediction))
print('\tPrediction:\n', prediction, '\n')


# EXTRAPOLATION 
# Shuffling is not done, and the data is sequentially split.
# The prediction is done on the values which will happen in the future. 
print('Extrapolation:\n')
train_x, temp_x, train_y, temp_y = train_test_split(X3, Y, test_size=0.2, random_state=42, shuffle=False)
test_x, val_x, test_y, val_y = train_test_split(temp_x, temp_y, test_size=0.7, random_state=42, shuffle=False)
model = LinearRegression(fit_intercept=False).fit(train_x, train_y)
prediction = model.predict(test_x)
print('\tLoss:\t', Linear_Regression.SSE(test_y, prediction))
print('\tPrediction:\n', prediction)

print('\nEND OF ASSIGNMENT 2\n')
print('\n#############################################################################################\n')
#########################################################################################################################################################