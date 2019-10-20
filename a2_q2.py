
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):

    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x,y, tau,lam=1e-5):

    up = np.exp(-l2(np.array([test_datum]), x)/(2*(tau**2)))
    down = np.exp(logsumexp(-l2(np.array([test_datum]), x)/(2*(tau**2))))
    a = up/down
    a = a.reshape(a.shape[1],)
    A = np.diag(a)
    
    ## calculate w
    w = np.linalg.solve(np.linalg.multi_dot([x.T, A, x])+lam*np.diag(np.ones(d)), np.linalg.multi_dot([x.T, A, y]))
    ##calculate output
    y_hat = np.dot(test_datum, w)
    return y_hat

def run_validation(x,y,taus,val_frac):

    ## calculate train losses
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=val_frac, random_state=42)   
    
    loss_train_ave = np.empty(len(taus))
    loss_val_ave = np.empty(len(taus))
    for num in range(len(taus)):
        predict_train = np.empty(len(x_train))
        loss_train = np.empty(len(x_train))
        for i in range(len(x_train)):
            predict_train[i] = LRLS(x_train[i], x_train, y_train, taus[num], lam=1e-5)
        loss_train = 0.5*(predict_train - y_train)**2
        loss_train_ave[num] = np.average(loss_train)
        
        ## calculate val loss
        predict_val = np.empty(len(x_test))
        loss_val = np.empty(len(x_test))
        for j in range(len(x_test)):
            predict_val[j] = LRLS(x_test[j], x_train, y_train, taus[num], lam=1e-5)
        loss_val = 0.5*(predict_val - y_test)**2
        loss_val_ave[num] = np.average(loss_val)
        
        
    return loss_train_ave, loss_val_ave


if __name__ == "__main__":
#    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    plt.figure(figsize=(5, 15))
    plt.subplot(3,1,1) 
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, train_losses, label='train')
    plt.semilogx(taus, test_losses, label ='test')
    plt.title('Losses with tau in range [10,1000]')
    plt.xlabel('Tau')
    plt.ylabel('Loss')
    plt.legend()


    #TO SET TAU TO LARGE NUMBER CLOSE TO INFINITE  
    plt.subplot(3,1,2) 
    taus_large = np.logspace(1.0, 20, 100)
    train_losses_l, test_losses_l = run_validation(x,y,taus_large,val_frac=0.3)
    plt.semilogx(taus_large, train_losses_l, label='train')
    plt.semilogx(taus_large, test_losses_l, label ='test')
    plt.title('Losses with tau near to infinity')
    plt.xlabel('Tau')
    plt.ylabel('Loss')
    plt.legend()


    #TO SET TAU TO SMALL NUMBER CLOSE TO 0
    plt.subplot(3,1,3) 
    np.seterr(divide='ignore', invalid='ignore')
    taus_small = np.logspace(-2, 3, 100)
    train_losses_s, test_losses_s = run_validation(x,y,taus_small,val_frac=0.3)
    plt.semilogx(taus_small, train_losses_s, label='train')
    plt.semilogx(taus_small, test_losses_s, label='test')
    plt.title('Losses with tau near to zero')
    plt.xlabel('Tau')
    plt.ylabel('Loss')
    plt.legend()


