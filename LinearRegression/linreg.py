import math
import numpy as np

LARGE_INTEGER = 9999999

# SSE
def loss_function(w,b,X,T,N):
    summation = 0
    i = 0
    for x,t in zip(X,T):
        summation +=(t - b - (w*x))**2
    return summation/(N*2)

# calculates the gradient of the loss function for the update equation
def loss_gradient(w,b,learning_rate,X,T,N):
    dl_db = intercept(w,b,X,T,N)*learning_rate
    dl_dw = coefficients(w,b,X,T,N)*learning_rate
    return dl_db, dl_dw
    
def intercept(w,b,X,T,N):
    sum = 0.0
    for x,t in zip(X,T):
        sum +=(t - b - (w*x))
    return -sum/N

def coefficients(w,b,X,T,N):
    sum = 0.0
    for x,t in zip(X,T):
        sum +=(t - b - (w*x))*x
    return -sum/N

def gradient_descent(X,T,N):
    w = 0.0 # first guess
    b = 0.0 
    previous_loss = LARGE_INTEGER
    iteration = 0
    learning_rate = 0.1
    loss = loss_function(w,b,X,T,N)
    converged = False
    while loss > 0.001 and not converged:
        loss = round(loss_function(w,b,X,T,N),7)
        print("at iteration ",iteration, " Loss: ",loss )
        if previous_loss == loss:
            print("Converged at iteration:", iteration, " Loss: ",loss )
            converged = True
        old_b,old_w = b,w
        b_gradient,w_gradient = loss_gradient(old_w,old_b,learning_rate,X,T,N)  # gradients of b and w
        b = round(old_b-b_gradient,4)
        w = round(old_w-w_gradient,4)
        previous_loss = loss
        iteration+=1
    return b,w

def linear_regression(X,T):
    N = float(len(X))
    b,w = gradient_descent(X,T,N)
    print(f'Linreg Equation: y = {b} + {w}x')




def main():
    # dummy data 
    X = [0.5, 1.4,2]
    T = [1.0, 1.9,3] 
    linear_regression(X,T) 


if __name__ == '__main__':
  main()



        


        









     






