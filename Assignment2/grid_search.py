import math
import numpy as np 
import matplotlib.pyplot as plt
from LinearRegression_Self import *

def grid_search(df, start_angle, end_angle, step_size): 
    
    theta_deg = [i for i in range(start_angle, end_angle+1, step_size)]
    theta_rad = [math.radians(i) for i in range(0, 61, 5)]

    m_tan_opt = []
    for i in range(0, len(theta_rad)):
        m_tan_opt.append([math.tan(theta_rad[i])])

    X = np.expand_dims(df.x, 1)
    Y = np.array(df.y)

    tan_error = []
    min_angle = 0
    min_error = 10000000
    for i in range(0, len(theta_rad)):
        predictions = Linear_Regression.predict(m_tan_opt[i], X)
        error = Linear_Regression.SSE(Y, predictions)
        tan_error.append(error)
        min_error = min(min_error, tan_error[i])

    for i in range(0, len(tan_error)): 
        if(tan_error[i] == min_error): 
            min_angle = theta_deg[i]

    plt.plot(tan_error, theta_deg, 'r-')
    plt.xlabel('SSE')
    plt.ylabel('Angle in degrees') 
    plt.title('Grid Search: Angle vs SSE')
    plt.legend(['Angle (degrees)', 'SSE'], loc = 'lower right')
    plt.grid()
    plt.show()
    plt.savefig('Grid_search.png')       
    plt.close()


    return min_error, min_angle #min(tan_error) 
    pass
# 40000 - 160000