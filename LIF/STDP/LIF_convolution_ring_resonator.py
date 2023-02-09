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

from LIF_solvers import LIF_RK
from LIF_layers import layers_conv, layer_pool, lateral_inihition, stdp_competition
from functions import stimuli_gen, plot, weight_ring, conv_ring, temporal_decoding



#%% Load data 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

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
Layer_1 = layers_conv(k=3,filters=10, data=d, dt=dt, n_points=N_points, time=time, ring_option=1, C=0.00005)
x1 = Layer_1.conv2D(vis=False)

Layer_1_ = lateral_inihition(data=x1, area_size=5, n_points =N_points )
x1_ = Layer_1_.wta(vis=True)

Layer_1__ = stdp_competition(data=x1, area_size=5, n_points =N_points )
x1__ = Layer_1__.wta(vis=False)

Layer_2 = layer_pool(data=x1, time=time, dt=dt, n_points=N_points, op='max', stride=2, pool_size=2)
x2 = Layer_2.Pool2D(vis=True)

Layer_3 = layers_conv(k=3,filters=10, data=x2, dt=dt, n_points=N_points, time=time, ring_option=1, C=0.00005)
x3 = Layer_3.conv2D(vis=True)

Layer_4 = lateral_inihition(data=x3, area_size=5, n_points =N_points )
x4 = Layer_4.wta(vis=True)





