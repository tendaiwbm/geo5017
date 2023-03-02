# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 06:33:44 2023

@author: 3Q
"""

import matplotlib.pyplot as plt
import numpy
import math
import sys

# objective function
def mse(y,y_bar):
    return sum([(p - y_bar[i])**2 for i,p in enumerate(y)]) / len(y)

# initial model X = a2*t**2 + a1*t + a0
def fX(t,a2,a1,a0):
    return a2*(t**2) + a1*t + a0
    
# partial derivative of mse w.r.t alpha2
def dAplha2(t,x,x_pred):
    return t**2 * (x - x_pred)   
 
# partial derivative of mse w.r.t alpha1
def dAlpha1(t,x,x_pred):
    return t * (x - x_pred)

# partial derivative of mse w.r.t alpha0
def dAlpha0(x,x_pred):
    return x - x_pred

# evaluates the gradient of each partial derivative
# re-used for alpha2, alpha1 and alpha0
def gradient(n,dAlpha):
    return (-2/n) * sum(dAlpha)


def gradient_descent(X,t,learn_rate,coeffs):
    
    X_pred = [fX(ti,*coeffs) for ti in t]
    partials_alpha2 = [dAplha2(t[i], X[i], X_pred[i]) for i,x in enumerate(X)]
    partials_alpha1 = [dAlpha1(t[i], X[i], X_pred[i]) for i,x in enumerate(X)]
    partials_alpha0 = [dAlpha0(X[i], X_pred[i]) for i,x in enumerate(X)]
    
    print(f'''partials_aplha2\t{partials_alpha2}
partials_alpha1\t{partials_alpha1}
partials_alpha0\t{partials_alpha0}''')
    
    # update model coefficients
    coeffs[0] = coeffs[0] - learn_rate * gradient(len(partials_alpha2),partials_alpha2)
    coeffs[1] = coeffs[1] - learn_rate * gradient(len(partials_alpha1),partials_alpha1)
    coeffs[2] = coeffs[2] - learn_rate * gradient(len(partials_alpha0),partials_alpha0)
    
    print('coeffs\t',coeffs)
    
    X_pred = [fX(ti,*coeffs) for ti in t]
    loss = mse(X,X_pred)
    
    print('X\t',X)
    print('X-predicted\t',X_pred)
    print('loss\t',loss)
    
    return loss,coeffs,X_pred


def regress(X,t,coeffs,max_iter,rate,tol):
    
    losses = []
    alpha2 = None
    alpha1 = None
    alpha0 = None
    i = 0

    while i < max_iter:
        
        loss,new_coeffs,X_pred = gradient_descent(X,t,rate,coeffs)
        losses.append(loss)
        
        deltaA2 = numpy.abs(coeffs[0] - new_coeffs[0])
        deltaA1 = numpy.abs(coeffs[1] - new_coeffs[1])
        deltaA0 = numpy.abs(coeffs[2] - new_coeffs[2])
        
        if all(delta < tol for delta in [deltaA0,deltaA1,deltaA2]):
            alpha2 = coeffs[0]
            alpha1 = coeffs[1] 
            alpha0 = coeffs[2]
         
        
        i += 1
    
    coeffs = [alpha2,alpha1,alpha0]
    return X_pred,coeffs
        

def main():
    
    # Polynomial regression on X-coordinate only
    # t = time value for each x-coordinate
    # coeffs = random coefficients a2,a1 and a0 for the initial model
    # initial model >> X = a2*t**2 + a1*t + a0; see function fX
    X = [2,1.08,-0.83,-1.97,-1.31,0.57]
    t = [1,2,3,4,5,6]
    original_coeffs = [9,-6,2]
    coeffs = [9,-6,2]
    max_iter = 10000
    rate = 0.0001
    tol = 0.01
    
    # final prediction, X_pred obtained after minimising sum of squares
    X_pred,final_coeffs = regress(X,t,coeffs,max_iter,rate,tol)
    print('original coeffs\t',original_coeffs)
    print('final coeffs\t',final_coeffs)
   
    plt.figure(figsize=(20,10))
    plt.plot(t,X,marker='.',markersize=20,label='original')
    plt.plot(t,X_pred,marker='s',label='final_prediction')
    plt.legend(loc='lower right')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.savefig('output.png')
    plt.show()
    
    
    
    
    
    


if __name__ == '__main__':
    main()
    
    
    
