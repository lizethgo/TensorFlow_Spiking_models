#!/usr/bin/env python3
"""
File        : LIF_layers.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : This file contains the classes for the following layers:
             
References: https://www.researchgate.net/publication/322568485_Spike_Neural_Models_Part_II_Abstract_Neural_Models
"""

import numpy as np
import matplotlib.pyplot as plt
from functions_tf import conv_ring, weight_ring
from SNN_test2_tf import LIF_euler
import tensorflow as tf



def plot_spikes(data, title):
    scatter_plot = np.reshape(data, (np.shape(data)[-2]*np.shape(data)[-2], np.shape(data)[-1]))
    indexes_filters = np.where(scatter_plot>20)
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(indexes_filters[1][:], indexes_filters[0][:], s=0.5, color='red')
    #plt.ylim(0,784)
    plt.title(title)
    plt.show()
    
def plot_features(data, title):
    feature_map = np.reshape(data,(int(np.sqrt(np.shape(data)[-1])),int(np.sqrt(np.shape(data)[-1]))))
    feature_map = 255*feature_map  ## ready to print
    plt.imshow(feature_map)
    plt.title(title)
    plt.show()



def counting_spikes(data, size, time_points):
    #data_flatten = np.reshape(data,(size*size, np.shape(data)[-1]))
    data_flatten = data
    data_flatten = np.transpose(data_flatten)
    rank_indexes = np.where(data_flatten>28)

    feature_map = np.zeros([1,size*size])
    a,ind = np.unique(rank_indexes[1][:],return_index=True)
    b =rank_indexes[0][ind]
    a = np.reshape(a,(1,np.shape(a)[0]))
    b = np.reshape(b,(1,np.shape(b)[0]))  ### N_points where a > 20
    if np.size(b) > 0:
        t_1 = np.min(b)  ## first spike
        for i in range(np.shape(b)[1]):
            feature_map[0,a[0,i]] = np.abs(((b[0,i]-t_1)/time_points)-1)
        n_spikes = np.sum(feature_map)

    else:
         n_spikes=0
        
    return  n_spikes
    



class layer_conv_tf:
    
    def __init__(self, k, data, filters, time, dt, n_points, ring_option, C, **kwargs):
        self.k = k
        self.data = data
        self.time = time
        self.dt = dt
        self.n_points = n_points
        self.filters = filters
        self.filter_size = np.shape(data)[-2]-k #TODO: Calaculate it automatically
        self.ring_option = ring_option
        self.C=C
        self.dW = np.zeros((1,int(time/dt)))
        
    def temporal_decoding(self, input_map):
        
        """
        For visualization purposes only.
        Input: 
            1. A feature map of size [filter_size, filter_size, n_points]
            2. vis. If True, it will print the decoded feature map
        Output: Decoded feature map.
        """
        # #data_flatten = np.reshape(input_map,(self.filter_size*self.filter_size, np.shape(input_map)[-1]))
        # data_flatten = tf.reshape(input_map,shape=(28*28, np.shape(input_map)[-1]))
        # print(tf.shape(data_flatten))
        # data_flatten = tf.transpose(data_flatten)
        # print(tf.shape(data_flatten))
        # rank_indexes = tf.where(data_flatten>0)
        
        # feature_map = tf.zeros(shape=[1,28*28])
        # print(tf.shape(feature_map))
        
        # a,ind = tf.unique(rank_indexes[1,:])
        # print(ind)
        # b =rank_indexes[0,ind.numpy()]
        # a = tf.reshape(a,shape=(1,np.shape(a)[0]))
        # b = tf.reshape(b,shape=(1,np.shape(b)[0]))  ### N_points where a > 20
        # t_1 = tf.math.minimum(b)  ## first spike
        
        # for i in range(np.shape(b)[1]):
        #     feature_map[0,a[0,i]] = tf.math.abs(((b[0,i]-t_1)/self.n_points)-1)
        # print_ = True
        # if print_:
        #     for i in range(self.filters):
        #         plot_features(feature_map, title='Feature Maps')
                
        data_flatten = np.reshape(input_map,(28*28, 9))
        data_flatten = np.transpose(data_flatten)
        rank_indexes = np.where(data_flatten>20)
        
        feature_map = np.zeros([1,28*28])
        a,ind = np.unique(rank_indexes[1][:],return_index=True)
        b =rank_indexes[0][ind]
        a = np.reshape(a,(1,np.shape(a)[0]))
        b = np.reshape(b,(1,np.shape(b)[0]))  ### N_points where a > 20
        t_1 = np.min(b)  ## first spike
        
        for i in range(np.shape(b)[1]):
            feature_map[0,a[0,i]] = np.abs(((b[0,i]-t_1)/self.n_points)-1)
        print_ = True
        if print_:
            for i in range(self.filters):
                plot_features(feature_map, title='Feature Maps')
        
        
#     def conv_op(self, vis, p):
#         """
#         2D convolutional operation
#         Returns: a matrix fo size [#_filters, filter_size, filter_size]
#         It performs 2D convololution over self.data using the ring resonator model
#         """
#         LIF = LIF_RK(self.C)
#         self.filter_size = np.shape(self.data)[-2]-self.k
#         intensity = np.zeros([self.filter_size,self.filter_size,self.n_points])
#         p_test = 0
        
        
#         for i in range(self.filter_size):
#             for j in range(self.filter_size):
#                 temp2 =  np.zeros([self.n_points])
#                 for ch in range(1): # input channels
#                     #print(np.random.randint(1,11))
#                     stimu=conv_ring(self.data[0,i:i+self.k,j:j+self.k,:],self.k, option=self.ring_option)
#                     #print(np.shape(self.time), np.shape(stimu), np.shape(self.dt) )
#                     temp1, p_test, m, dW_temp = LIF.single_neuron(time=self.time, stimuli=stimu, dt=self.dt, stdp=False, p=p)
#                     temp2 = temp1 + temp2
                    
                
#                 if i ==0 :
#                     if j == 0:
#                         print('delat W of filter 0')
#                         plt.plot(dW_temp)
#                         plt.plot(m)
#                         plt.plot(p_test)
#                         plt.show()
#                 self.dW=dW_temp
#                 #intensity[i,j,:]=LIF.single_neuron(time=self.time, stimuli=stimu, dt=self.dt)
#                 intensity[i,j,:]=temp2
#         #print('intensity', np.shape(intensity))

#         if vis: plot_spikes(intensity, title='Feature Maps')
#         return intensity, p_test
    
# # v1, p1, m1, dW1 = LIF.single_neuron(stimuli=I_, dt=dt, time=time, stdp=False, p=p)
    
    def conv2D(self, vis):
        """
        It performs conv2D operations to generate filters. The resulting number of filter is self.filters
        Returns L_map, which is a 4D matrix of size [# filters, filter_size, filter_size, N_points]
        """
        LIF = LIF_euler(0.01, 5)
        p = tf.zeros((1,500))
        L_map = tf.zeros(shape=(1, 28, 28, 10, 500), dtype=tf.double)
        
        
        weights =  weight_ring(V=0.01*tf.random.uniform(shape=(3,3, 1, self.filters )))
        data_in = tf.reshape(self.data[:,:,:,0], shape=(1,28,28,1))
        aux = 0
        
        for i in range(500):
            L = tf.nn.conv2d(input=data_in, filters=weights, strides=1, padding='SAME', data_format='NHWC', dilations=None, name=None)  
            

            L = tf.reshape(L, shape=[1,28,28,10,1])
            if i == 1:
                L_ = L
            if i>1:
                L_ = tf.concat([L_,L], axis = -1)
                
         
            ##### SNN
        # L_ = tf.cast(L_, dtype=tf.float64)        
        # test = np.zeros((10,28,28,499))
        # for k in range(28):
        #     for p in range(28):
        #             test1 = LIF.solve_single_neuron(L_[0,k,p,:])
        #             #print(np.shape(test1))
        #             test[0,k,p,:]= LIF.solve_single_neuron(L_[0,k,p,:])

        
        print(tf.shape(L_))
            
        #print(np.shape(L))
        # for i in range(0,self.filters):
        #     L_map[i,:,:,:], p_test=self.conv_op(vis=vis, p=p)
        #     p = p_test
        # #print(np.shape(L_map))
        # #vis = True
        if vis:
            L_ = tf.reshape(L_, shape=(10,28,28,499))
            data = L_.numpy()
            for i in range(self.filters):
                self.temporal_decoding(input_map=data[0,:,:,:])
        return L_
    




        

        
    
