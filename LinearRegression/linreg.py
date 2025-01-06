import math
import numpy as np
import pandas as pd

LARGE_INTEGER = 9999999

class LinearRegression:

    def __init__(self):
        self.w = None
        self.b = None

    def train(self,input,output):
        X = input
        T = output

        self.b,self.w = self.gradient_descent(X,T)
        print(f'LinReg Equation: y = {self.b} + {self.w}x')
    
    def predict(self,input_data):
        y = [self.b+self.w*x for x in input_data]
        print(f"Predicted: {y}")
        return y

    # SSE
    def loss_function(self,w,b,X,T,N):
        summation = 0
        i = 0
        for x,t in zip(X,T):
            summation +=(t - b - (w*x))**2
        return summation/(N*2)

    # mse
    def score(self,X,T):
        N = float(len(X))
        mse = self.loss_function(self.w,self.b,X,T,N)/N
        print(f"MSE: {mse}")
        return mse


    # calculates the gradient of the loss function for the update equation
    def loss_gradient(self,w,b,learning_rate,X,T,N):
        dl_db = self.intercept(w,b,X,T,N)*learning_rate
        dl_dw = self.coefficients(w,b,X,T,N)*learning_rate
        return dl_db, dl_dw

    def intercept(self,w,b,X,T,N):
        sum = 0.0
        for x,t in zip(X,T):
            sum +=(t - b - (w*x))
        return -sum/N

    def coefficients(self,w,b,X,T,N):
        sum = 0.0
        for x,t in zip(X,T):
            sum +=(t - b - (w*x))*x
        return -sum/N

    def gradient_descent(self,X,T):
        w = 0.0 # first guess
        b = 0.0 
        N = float(len(X))
        previous_loss = LARGE_INTEGER
        iteration = 0
        learning_rate = 0.1
        loss = self.loss_function(w,b,X,T,N)
        converged = False
        while loss > 0.001 and not converged:
            loss = round(self.loss_function(w,b,X,T,N),5)
            if previous_loss == loss:
                print("Converged at iteration:", iteration, " Loss: ",loss )
                converged = True
            #print("At iteration: ",iteration, " loss: ",loss )
            old_b,old_w = b,w
            b_gradient,w_gradient = self.loss_gradient(old_w,old_b,learning_rate,X,T,N)  # gradients of b and w
            b = round(old_b-b_gradient,4)
            w = round(old_w-w_gradient,4)
            previous_loss = loss
            iteration+=1
        return b,w

        


def main():
    X_train = [0.5, 1.4,2,0.6,2,2.1]
    y_train = [1.0, 1.9,3,1.1,3,3.1] 
    X_test = [0.6,2,2.5]
    y_test = [1.2,3.1,3.3]
    model = LinearRegression()
    model.train(X_train,y_train)
    print(f"Actual:  {y_test}")
    y_pred = model.predict(X_test)
    model.score(y_test,y_pred)
    


if __name__ == '__main__':
  main()



        


        









     






