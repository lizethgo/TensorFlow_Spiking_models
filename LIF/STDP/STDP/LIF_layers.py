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
from functions import conv_ring
from LIF_solver_STDP import LIF_RK


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
    



class layers_conv:
    
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
        data_flatten = np.reshape(input_map,(self.filter_size*self.filter_size, np.shape(input_map)[-1]))
        data_flatten = np.transpose(data_flatten)
        rank_indexes = np.where(data_flatten>20)
        
        feature_map = np.zeros([1,self.filter_size*self.filter_size])
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
        
        
    def conv_op(self, vis, p):
        """
        2D convolutional operation
        Returns: a matrix fo size [#_filters, filter_size, filter_size]
        It performs 2D convololution over self.data using the ring resonator model
        """
        LIF = LIF_RK(self.C)
        self.filter_size = np.shape(self.data)[-2]-self.k
        intensity = np.zeros([self.filter_size,self.filter_size,self.n_points])
        p_test = 0
        
        
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                temp2 =  np.zeros([self.n_points])
                for ch in range(1): # input channels
                    #print(np.random.randint(1,11))
                    stimu=conv_ring(self.data[0,i:i+self.k,j:j+self.k,:],self.k, option=self.ring_option)
                    #print(np.shape(self.time), np.shape(stimu), np.shape(self.dt) )
                    temp1, p_test, m, dW_temp = LIF.single_neuron(time=self.time, stimuli=stimu, dt=self.dt, stdp=False, p=p)
                    temp2 = temp1 + temp2
                    
                
                if i ==0 :
                    if j == 0:
                        print('delat W of filter 0')
                        plt.plot(dW_temp)
                        plt.plot(m)
                        plt.plot(p_test)
                        plt.show()
                self.dW=dW_temp
                #intensity[i,j,:]=LIF.single_neuron(time=self.time, stimuli=stimu, dt=self.dt)
                intensity[i,j,:]=temp2
        #print('intensity', np.shape(intensity))

        if vis: plot_spikes(intensity, title='Feature Maps')
        return intensity, p_test
    
# v1, p1, m1, dW1 = LIF.single_neuron(stimuli=I_, dt=dt, time=time, stdp=False, p=p)
    
    def conv2D(self, vis):
        """
        It performs conv2D operations to generate filters. The resulting number of filter is self.filters
        Returns L_map, which is a 4D matrix of size [# filters, filter_size, filter_size, N_points]
        """
        p = np.zeros((1,500))
        L_map = np.zeros((self.filters, self.filter_size, self.filter_size, self.n_points))
        for i in range(0,self.filters):
            L_map[i,:,:,:], p_test=self.conv_op(vis=vis, p=p)
            p = p_test
        #print(np.shape(L_map))
        #vis = True
        if vis:
            for i in range(self.filters):
                self.temporal_decoding(input_map=L_map[1,:,:,:])
        return L_map
    

class layer_pool:
    def __init__(self, data, time, dt, n_points, op, stride, pool_size, **kwargs):
        self.data = data
        self.time = time
        self.dt = dt
        self.n_points = n_points
        self.op = 'max'
        self.stride = stride
        self.pool_size = pool_size
        self.data_size = np.shape(data)[-2] # filter size TODO: Calaculate it automatically
        
    def temporal_decoding(self, input_map):
        
        """
        For visualization purposes only.
        Input: 
            1. A feature map of size [#feature_maps, data_size, data_size, n_points]
            2. vis. If True, it will print the decoded feature map
        Output: Decoded feature map.
        """
        pool_size = np.shape(input_map)[-2]
        data_flatten = np.reshape(input_map,(pool_size*pool_size, np.shape(input_map)[-1]))
        data_flatten = np.transpose(data_flatten)
        rank_indexes = np.where(data_flatten>20)
        
        feature_map = np.zeros([1,pool_size*pool_size])
        a,ind = np.unique(rank_indexes[1][:],return_index=True)
        b =rank_indexes[0][ind]
        a = np.reshape(a,(1,np.shape(a)[0]))
        b = np.reshape(b,(1,np.shape(b)[0]))  ### N_points where a > 20
        t_1 = np.min(b)  ## first spike
        for i in range(np.shape(b)[1]):
            feature_map[0,a[0,i]] = np.abs(((b[0,i]-t_1)/self.n_points)-1)
        print_ = True
        if print_: plot_features(feature_map, title='Pooling')


        
    def Pool2D(self, vis=True):
        ### additional parameters 
        a = (int(self.data_size/self.stride)-1)*self.stride 
        b =  np.mod((int(self.data_size/self.stride)*self.stride+self.pool_size-1),self.data_size)
        lp = a - b
        padding = int(self.data_size/self.stride)*self.stride+self.pool_size-self.data_size  #TODO: Implement padding as an optional argument
        pooling = np.zeros([np.shape(self.data)[0], int(lp/self.stride),int(lp/self.stride),self.n_points+1])
    
        p1 = 0
        p2 = 0
        
        for f in range (0, np.shape(self.data)[0]):
            for t in range(0, np.shape(self.data)[-1]):
                for i in range(0,lp,self.stride,):
                    for j in range(0,lp,self.stride):
                            pooling[f,p1,p2,t]=np.max(self.data[f,i:i+self.stride, j:j+self.stride,t]) 
                            p2 += 1
                    p2 = 0
                    p1 += 1
                p1 = 0

        if vis == True:
            plot_spikes(pooling[1,:,:,:], title='Pooling')
            self.temporal_decoding(input_map=pooling[0,:,:,:])
        return pooling
        


        
class lateral_inihition:
    
    def __init__(self, data, area_size, n_points, **kwargs):
        self.data = data
        self.area_size = area_size
        self.n_points = n_points
        self.filters = np.shape(data)[0] # number of filters
        
    def wta(self,vis):
        ### auxiliary vars
        counts = np.zeros(self.filters) 
        add_before_li = np.zeros([np.shape(self.data)[1],np.shape(self.data)[1]])
        add_after_li = np.zeros([np.shape(self.data)[1],np.shape(self.data)[1]])     
        mask = np.zeros([self.filters,1])
        n_filters = np.shape(self.data)[1]
        out = np.copy(self.data)
        
        for i in range(0, n_filters):
            for j in range(0, n_filters):
        
                #for m in range(0,maps):
                temp = self.data[0,i,j:j+1,:]
                for m_ in range(0,self.filters-1):
                    temp = np.concatenate((temp,self.data[m_,i,j:j+1,:]),axis=0)
                    counts[m_] = counting_spikes(self.data[m_,i,j:j+1,:], size=n_filters, time_points=self.n_points)

                add_before_li[i,j] = np.sum(counts)
                max_coo = np.unravel_index(np.argmax(temp), np.shape(temp))
                mask[max_coo[0],:] = 1
                temp=temp*mask ## lateral inhibition 
                ## applying latera inhibition
                for m_ in range(0, self.filters):
                    out[m_,i,j:j+1,:] = self.data[m_, i,j:j+1,:]*mask[m_-1,:]
                    counts[m_-1] = counting_spikes(out[m_,i,j:j+1,:], size=n_filters, time_points=self.n_points)
                    ## [i,j:j+1,:] -> j:j+1 just to make compatible the sizes for the function counting spikes
                add_after_li[i,j] =  np.sum(counts) # just to visualize lateral inhibition
                mask[max_coo[0],:] = 0
        if vis:
            plt.imshow(add_before_li)
            plt.show()
            plt.imshow(add_after_li)  
            plt.show()   
        return out
        
    
class stdp_competition:
        def __init__(self, data, area_size, n_points, **kwargs):
            self.data = data
            self.area_size = area_size
            self.n_points = n_points
            self.filters = np.shape(data)[0] # number of filters
            
        def wta(self,vis):

            mask = np.zeros([self.area_size,self.area_size])
            data_ = self.data.copy()
            STDP_competition = np.zeros([np.shape(self.data)[1],np.shape(self.data)[1]])
            out = self.data.copy()
            n_filters = np.shape(self.data)[1]
  
    
            for t in range(0, self.n_points):
                for i in range(0, int(n_filters/self.area_size)):
                    for j in range(0, int(n_filters/self.area_size)):
                        for m_ in range(0, self.filters):
                            temp  = np.argmax(data_[m_, i*self.area_size:i*self.area_size+self.area_size,j*self.area_size:j*self.area_size+self.area_size,t])
                            max_coo = np.unravel_index(np.argmax(temp), np.shape(data_[m_, i*self.area_size:i*self.area_size+self.area_size,j*self.area_size:j*self.area_size+self.area_size,t]))
                            mask[max_coo[0], max_coo[1]]= 1
                            #if t ==0:
                               # print(max_coo[0], max_coo[1])
                            out[m_, i*self.area_size:i*self.area_size+self.area_size,j*self.area_size:j*self.area_size+self.area_size,t] = self.data[m_, i*self.area_size:i*self.area_size+self.area_size,j*self.area_size:j*self.area_size+self.area_size,t]*mask
                            mask[max_coo[0], max_coo[1]]= 0
                            
            

            for i in range(0, n_filters):
                for j in range(0, n_filters):
                    
                    for m_ in range(0, self.filters):
                        STDP_competition[i,j] =  STDP_competition[i,j]+counting_spikes(out[m_,i,j:j+1,:], size=n_filters, time_points=5001)
                        ## [i,j:j+1,:] -> j:j+1 just to make compatible the sizes for the function counting spikes       
            if vis:      
                plt.imshow(STDP_competition)  
                plt.show()  
            
            return out
            
        


  
        
     
