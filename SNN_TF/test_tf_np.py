# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:02:37 2022

@author: 20195088
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import numpy as np

import time as mytime


size = 10000


start = mytime.time()
tf.random.uniform(shape=[size,1])/500
end = mytime.time() 

print("TF1 : The time of execution of above program is :", end-start)

start = mytime.time()
tf.Variable(np.random.uniform(size=(size,1)))
end = mytime.time() 

print("TF2 : The time of execution of above program is :", end-start)


start = mytime.time()
np.random.uniform(size=(size,1))/500
end = mytime.time() 

print("NP : The time of execution of above program is :", end-start)




##############################################################################
#### assigning
size = 100
start = mytime.time()
v = tf.zeros(size, dtype=tf.float32)
v = tf.Variable(v, dtype=tf.float32)
v[0].assign(1)

X = tf.ones((size,size)) 
Y = tf.ones((size,1))

W = tf.Variable(np.random.uniform(size=(size,size)), name="W", dtype="float32")

X_NP = np.ones((size,size))
Y_NP = np.ones((size,1))

W_NP = np.random.uniform(size=(size,size))

#Loss = tf.pow(tf.add(Y, -tf.matmul(X, W)), 2, name="Loss")

for i in range(size): 
    v[i].assign(1)
end = mytime.time() 
print("TF1 : The time of execution of above program is :", end-start)


i = tf.constant(0)
c = lambda i : tf.less(i, size)
#b = lambda i, X, W: (tf.add(i,1),tf.matmul(X, W))

start = mytime.time()


def b(i):
#      v[i].assign(2)
#      #int(i)
      tf.matmul(X, W)
      return lambda i: (tf.add(i,1),)
#      #v[1].assign(1)
#      #print(i)
#      #return i+1

#b = lambda i: (tf.add(i,1),)
tf.while_loop(c,b,[i])

end = mytime.time() 
print("TF1_1 : The time of execution of above program is :", end-start)



start = mytime.time()
v = tf.zeros(size, dtype=tf.float32)
v = tf.Variable(v)
v = tf.ones(size, dtype=tf.float32)

#v[0].assign(1)
#
#for i in range(100): 
#    v[i].assign(1)
#end = mytime.time() 
print("TF2 : The time of execution of above program is :", end-start)



start = mytime.time()
v = np.zeros(size)
v[0]=1
for i in range(size): 
    v[i]=3
    np.matmul(X_NP,W_NP)
end = mytime.time()     
print("NP1 : The time of execution of above program is :", end-start)


start = mytime.time()
v = np.zeros(size)
v = np.ones(size)
    
print("NP2 : The time of execution of above program is :", end-start)