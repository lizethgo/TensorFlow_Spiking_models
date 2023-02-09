# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:34:41 2022

@author: 20195088
"""

#!/usr/bin/env python3
"""
File        : single_neuron.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : Convolution operation of LIF model
"""

import numpy as np
import math
import matplotlib.pyplot as plt

from SNN_test2_tf import LIF_euler
from SNN_layers_tf import layer_conv_tf
from functions_tf import stimuli_gen, plot, weight_ring, conv_ring



#%% Load data 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# program to compute the time
# of execution of any python code
import time as mytime
 
# we initialize the variable start
# to store the starting time of
# execution of program
start = mytime.time()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0,:,:])
plt.show


step_length = 5
#N_points=5001-step_length
time = 5
dt = 0.01
N_points = int(time/dt)
x_train_t = x_train/255  ## indexes
x_train_t = np.abs(x_train_t-1)
x_train_t = N_points*(x_train_t) # calculating the delta t depending on the pixel intensity

d=np.zeros([np.shape(x_train)[1], np.shape(x_train)[1],N_points])
d_plot = np.zeros([np.shape(x_train)[1], np.shape(x_train)[1],int(N_points)+1])

for i in range(np.shape(x_train)[1]):
    for j in range(np.shape(x_train)[1]):
        d[i,j,int(x_train_t[0,i,j]):int(x_train_t[0,i,j])+step_length] = 1
        d_plot[i,j,int(x_train_t[0,i,j]):int(x_train_t[0,i,j])+1] = 1

        
d_plot = np.reshape(d_plot, (int(np.shape(x_train)[1]*np.shape(x_train)[1]),N_points+1)) 
indexes=np.where(d_plot>0.5)
fig = plt.figure(figsize=(6, 4))
plt.scatter(indexes[1][:], indexes[0][:], marker='|', s=15, color='red')
plt.show()   

d = np.reshape(d, (1,np.shape(d)[0], np.shape(d)[1], np.shape(d)[-1]))
#### Sequential model
Layer_1 = layer_conv_tf(k=3,filters=10, data=d, dt=dt, n_points=N_points, time=time, ring_option=1, C=0.00005)
x1 = Layer_1.conv2D(vis=False)

end = mytime.time()
 
# difference of start and end variables
# gives the time of execution of the
# program in between
print("The time of execution of above program is :", end-start)