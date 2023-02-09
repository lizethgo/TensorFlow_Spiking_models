# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:51:01 2022
Examnple of multi-task
@author: 20195088
"""

import tensorflow as tf
import numpy as np


import time as mytime



size = 1000


start = mytime.time()
X = tf.ones((size,size)) 
Y = tf.ones((size,1))

W = tf.Variable(np.random.uniform(size=(size,size)), name="W", dtype="float32")

Loss = tf.pow(tf.add(Y, -tf.matmul(X, W)), 2, name="Loss")
end = mytime.time() 

print("TF : The time of execution of above program is :", end-start)


start = mytime.time()
X_NP = np.ones((size,size))
Y_NP = np.ones((size,1))

W = np.random.uniform(size=(size,size))

Loss = np.power((Y-np.matmul(X,W)),2)


end = mytime.time() 

print("NP : The time of execution of above program is :", end-start)