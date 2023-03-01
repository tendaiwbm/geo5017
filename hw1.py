# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 06:33:44 2023

@author: 3Q
"""

import matplotlib.pyplot as plt
import numpy
import math
import sys


def mse(y,y_bar):
    return sum([(p - y_bar[i])**2 for i,p in enumerate(y)]) / len(y)


def fX(t,a0,b0,c0):
    a = a0*(t**2)
    b = b0*t
    c = c0
    return a+b+c
    

def dAplha2(t,x,x_pred):
    return t**2 * (x - x_pred)    

def dAlpha1(t,x,x_pred):
    return t * (x - x_pred)

def dAlpha0(x,x_pred):
    return x - x_pred


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
        
        if alpha0 and alpha1 and alpha2:
            break
        
        loss,new_coeffs,X_pred = gradient_descent(X,t,rate,coeffs)
        losses.append(loss)
        
        if numpy.abs(coeffs[0] - new_coeffs[0]) < tol:
            alpha2 = coeffs[0]
        elif numpy.abs(coeffs[1] - new_coeffs[1]) < tol:
            alpha1 = coeffs[1]   
        elif numpy.abs(coeffs[2] - new_coeffs[2]) < tol:
            alpha0 = coeffs[2]
       
        i += 1
    
    return X_pred
        

def main():
    
    # USING X-coordinate only
    # t = time value for each x-coordinate
    # coeffs = random coefficients a2,a1 and a0 for the initial model
    X = [2,1.08,-0.83,-1.97,-1.31,0.57]
    t = [1,2,3,4,5,6]
    coeffs = [5,6,5]
    max_iter = 10000
    rate = 0.0001
    tol = 0.01
    
    # final prediction, X_pred obtained after minimising sum of squares
    X_pred = regress(X,t,coeffs,max_iter,rate,tol)
   
    plt.figure(figsize=(20,10))
    plt.plot(t,X,marker='.',markersize=20,label='original')
    plt.plot(t,X_pred,marker='s',label='final_prediction')
    plt.legend(loc='lower right')
    plt.show()
    
    
    
    


if __name__ == '__main__':
    main()
    
    
    
sys.exit()
def plot_trajectory():
    
    x = [2,1.08,-0.83,-1.97,-1.31,0.57]
    y = [0,1.68,1.82,0.28,-1.51,-1.91]
    z = [1,2.38,2.49,2.15,2.59,4.32]

    figure = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x,y,z,'red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def d2func(t):
    return 12*t + 12 

def d1func(t):
    return 6*(t**2) + 12*(t) + 0.3

def func(t):
    return (2*(t**3), 4*(t**2), 0.3*t, 5)

def gradient_descent(x,grad,max_iter,rate,tolerance=0.01,cs=False,ca=False):
    prev_ = x
    j = 0
    while j < max_iter:
        j += 1
        print('prev\t\t',prev_)
        gr = grad(prev_) * rate
        if abs((gr)) < tolerance:
            if cs: # print(d1 and func wrt constant speed)
                print('velocity\t',d1func(gr))
                print('position\t',func(d1func(gr)))
                return func(d1func(gr)),d1func(gr)
            break
        
        next_ = prev_ - gr
        print('next\t\t',next_)
        
        prev_ = next_
    return func(d1func(prev_)),d1func(prev_)   
        
def magnitude(v):
    return math.sqrt(sum([i**2 for i in v]))

def magnitude2(v,p):
    return sum([i**p for i in v])

def objective(velocity,constants,observations):
    E = (velocity - (len(observations)*constants[3] + sum([magnitude(p) for p in observations])*constants[2] + sum([magnitude2(p,2) for p in observations])*constants[1] + sum([magnitude2(p,3) for p in observations])*constants[0]))**2
    return E   
        

def main():
    plot_trajectory()
    constants,velocity = gradient_descent(5, d2func, 10000, 0.01,cs=True)
    X = [(2,0,1),(1.08,1.68,2.38),(-0.83,1.82,2.49),(-1.97,0.28,2.15),
         (-1.31,-1.51,2.59),(0.57,-1.91,4.32)]
    print('SSE\t\t\t',objective(velocity,constants,X))
    
    #objective()
        
        
        


