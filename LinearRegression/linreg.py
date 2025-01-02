import math
import numpy as np


X = [0.5, 1.4,2]
T = [1.0, 1.9,3]
N = float(len(X))

# SSE
def loss_function(w,b):
    summation = 0
    i = 0
    for x,t in zip(X,T):
        summation +=(t - b - (w*x))**2
    return summation/(N*2)

# calculates the gradient of the loss function for the update equation
def loss_gradient(w,b,learning_rate):
    dl_db = intercept_partial(w,b)*learning_rate
    dl_dw = weights_partial(w,b)*learning_rate
    return dl_db, dl_dw
    
def intercept_partial(w,b):
    sum = 0.0
    for i in range(len(X)):
        sum +=(T[i] - b - (w*X[i]))
    return -sum/N

def weights_partial(w,b):
    sum = 0.0
    for i in range(len(X)):
        sum +=(T[i] - b - (w*X[i]))*X[i]
    return -sum/N

def gradient_descent():
    w = 0.0 # first guess
    b = 0.0 
    iteration = 0
    learning_rate = 0.01
    loss = loss_function(w,b)
    while loss >0.001 and iteration < 200:
        loss = loss_function(w,b)
        print("at iteration ",iteration)
        print("loss ", loss)
        old_b,old_w = b,w
        b_gradient,w_gradient = loss_gradient(old_w,old_b,learning_rate)  # gradients of b and w
        b = old_b-b_gradient
        w = old_w-w_gradient
        iteration+=1
    print(w)
    print(b)

def main():
    gradient_descent()

  
if __name__ == '__main__':
  main()



        


        









     






