import numpy as np # type: ignore
from numpy import linalg # type: ignore


# y = wx 
# Code from Scratch 
class Linear_Regression:
    def OLS_Solution(X: np.ndarray, y: np.ndarray) -> np.ndarray: 
        pseudo_inverse = linalg.inv(np.dot(X.T, X))
        w = np.dot(np.dot(pseudo_inverse, X.T), y)
        return w
    
    def predict(w, X) :
        if(len(X.shape) != 2): 
            X = np.expand_dims(X, 1)

        w = np.array(w)
        y_pred = np.matmul(X, w)
        return y_pred
    
    def SSE(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
        error = 0 
        for i in range(0, len(y_pred)): 
            error += (y[i] - y_pred[i])**2
        return error 
        
    def MSE(y: np.ndarray, y_pred: np.ndarray) -> float: 
        error = 0
        for i in range(0, len(y_pred)): 
            error += (y[i] - y_pred[i])**2 
        error = float(error/len(y_pred))
        return error   

    # def fit_curve(X, y, y_pred) :
    #     plt.plot(X, y, 'r-')
    #     plt.plot(X, y_pred, 'b-')
    #     plt.xlabel('Time')
    #     plt.ylabel('Spring Position')
    pass     