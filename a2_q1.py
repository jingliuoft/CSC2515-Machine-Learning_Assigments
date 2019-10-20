
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_boston
import numpy as np


# In[2]:


boston = load_boston()
boston_data = np.array(boston.data)
boston_data = np.hstack((np.ones(506).reshape(506,1), boston_data))
t = boston.target.reshape(506,1)
w = np.zeros(14).reshape(14,1)

def cal_cost(theta, X, t):
    m = len(t)
    p = 1
    predict = X.dot(theta)
    a = abs(predict -t)
    cost = 1/m * np.sum(np.where(a <= p, 1/2*np.square(a), p*(a-0.5*p)))  
    return cost


# In[3]:


# Assume: find the minimun point when the differences with the previous iteration is less than 0.001
def grad_descent (X, t, theta, learning_rate, i, p):
    m = len(t)
    cost_history = np.zeros(i)
    theta_history = np.zeros((i, 14))
    for it in range(i):
        predict = X.dot(theta)
        theta_history[it:] = theta.T
        cost_history[it] = cal_cost(theta, X, t)
        theta1 = theta - (learning_rate/m)*X.T.dot(np.where(abs(predict-t)<=p, (predict - t), p*np.sign(predict-t)))
        if cal_cost(theta, X, t) - cal_cost(theta1, X, t)<0.001:
            print ('Last iteration of', it, 'with minimum cost is', cal_cost(theta1, X, t))
            break
        else:
            print('iteration of', it,  'cost is', cal_cost(theta1, X, t))
        theta = theta1
   
    return theta, cost_history, theta_history
        


# In[6]:


# test on learning rate = 1e-5
lr = 1e-5
n_itr = 1000
X = boston_data
theta, cost_history, theta_history = grad_descent(X, t, w, lr, n_itr,1)

