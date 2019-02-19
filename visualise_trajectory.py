# Simple Neural Network that learns a double well function. Graphviz is used to visualise the trajectory in weight space.
# Extend by a fractal-like function


#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
#import torch
import numpy as np
import matplotlib
matplotlib.use('agg') # different backend so we don't need tkinter
import matplotlib.pyplot as plt


def dblWell(x): # double well
    E0=1
    E1=0.05
    center_x = 5
    
    x = x-center_x
    return E0*(E1*x**4-x**2)

def fractalLike(x, L=10): # fractal-like Fourier series from Ann. Stat. 34, 1636
    coeffs = [.21, 1.25, .61, .25, .13, .10, 1.16, .18, .12, .23, .21, .19, .37, .99, .36, .02, .06, .08, .09, .04]
    series = 0.0
    for i,coeff in enumerate(coeffs):
        series += coeff*np.sin(i*2.0*np.pi*x/L)
    return 2.0*series


if __name__ == "__main__":
    # sample from function
    N       = 15         # how many total sampled points
    N_train = int(0.8*N) # how many training points
    
    L = 10 # sample points between 0 and this value
    X = L*np.random.rand(N)
    
    samplingFunction = dblWell
    y = samplingFunction(X) # choose which function to sample from
    
    X_train = X[:N_train]
    y_train = y[:N_train]
    X_test = X[N_train:]
    y_test = y[N_train:]
    
    plotrange = np.linspace(0, L, num=120)
    plt.plot(plotrange, samplingFunction(plotrange), label='true fct.')
    plt.plot(X, samplingFunction(X), '.', label='training data')
    plt.ylim(-10,15)
    plt.savefig('tmp.jpg')
    
    