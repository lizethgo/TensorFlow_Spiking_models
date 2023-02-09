#!/usr/bin/env python3
"""
File        : LIF_solvers_STDP.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : This file contains the classes:
              1. LIF_RK tha solves the LIF model equation based on the Runge Kutta 4th order. 
                 This function also implements the STDP rule.
              2. LIF_Euler that solves the LIF model equation based on the Euler Method
             
References: https://www.researchgate.net/publication/322568485_Spike_Neural_Models_Part_II_Abstract_Neural_Models


    
"""
###############################################################################
## Libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from functions import weight_ring, conv_ring
###############################################################################

    
  
class LIF_RK:
    """
    v(t)/dt = (-v(t) + RI(t))/tm
    where tm = R*C
    
    if == v(t)>vth  -> v(t) = V_max and v(t+1) = V_r
    
    """ 
    def __init__(self, C, **kwargs):
        self.V_thr = 15 
        self.R = 3000 
        self.C = C  # By reducing this value, the time constant tm is reduced C=0.00005
        self.V_r = 0 # Resting potential
        self.V_max = 30 # Maximum values of an spike 
        self.V_p = -1 # initial value
        self.train = 0
        self.post_trace = 0
        self.pre_trace = 0
        self.dp = 0
        self.dW_pre = 0
        self.dW_post = 0
        self.dW_p = 0
        self.dW_n = 0
        
        
    def RungeKutta4(self, I, dt):
        dv_1 = self.f_v(self.V_p, I) * dt
        dv_2 = self.f_v(self.V_p + dv_1 * 0.5, I) * dt
        dv_3 = self.f_v(self.V_p + dv_2 * 0.5, I) * dt
        dv_4 = self.f_v(self.V_p + dv_3, I) * dt
        dv = 1 / 6 * (dv_1 + dv_2 * 2 + dv_3 * 2 + dv_4)
        self.V_p += dv
    
    def f_v(self, v, I): 
        tm = (self.R * self.C)
        return (-v + self.R * I) / tm

    def solve(self, I, dt, stdp, p): 
        

        ### If the membrane potential v is greater than the threshold
        ### returns to the resting potential V_r
        if (self.V_p >= self.V_thr):
            self.V_p = self.V_r
            self.train = 0  ## to exclusuvely observe the spikes 

        else: 
            self.train =0 ## to exclusuvely observe the spikes 
            self.RungeKutta4(I, dt)
            self.post_trace = self.post_trace - ( dt / 0.05) * self.post_trace
            self.pre_trace =  self.pre_trace  - ( dt / 0.05) * self.pre_trace
            
        ### If the new value of the membrane potential v is greater than the threshold
        ### it generates an spike of mag 40
        if self.V_p >= self.V_thr: 
            self.V_p = self.V_max
            self.train = 30
            ### stdp
            
            
            self.post_trace = self.post_trace - (0.008*1.1)
            self.pre_trace =  self.pre_trace + 0.008
            
            self.dW_pre = self.pre_trace
            self.dW_post = self.post_trace
        
           
       
    
    def single_neuron(self, stimuli, dt, time, stdp, p):
        
        n_neurons = 1
        steps = int(time/dt)
        p_next =  np.zeros(steps)
        m =  np.zeros(steps)
        v = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        
        ### for STDP
        delta_w_p =  np.zeros((n_neurons,steps))
        delta_w_n =  np.zeros((n_neurons,steps))
        dW =  np.zeros((n_neurons,steps))
        dW_prev = 0
        delta_w_n_prev = 0
        delta_w_p_prev = 0
        stimuli = np.resize(stimuli, (1, np.shape(stimuli)[-1]))
        weigths = 1
        
        
        if stdp: 
            p = np.resize(p, (1, np.shape(p)[-1]))
            weigths = 3*np.random.randint(100)/50000

            
        for i in range(steps):
            weigthed_sum = weigths*stimuli[:,i]
            self.solve(weigthed_sum, dt, stdp=False, p=p)
            v[i] = self.V_p
            p_next[i] = self.pre_trace
            m[i] = self.post_trace

            #delta_w_p[i] = self.dW_p + self.dW_post
            #delta_w_n[i] = self.dW_n + self.dW_pre
            
            if stdp:
                delta_w_p[:,i], delta_w_n[:,i], dW[:,i] = self.stdp_function(m, n_neurons, stimuli, v, p, i, steps, dW_prev,delta_w_n_prev, delta_w_p_prev )
                dW_prev = np.resize(dW[:,i], (np.shape(dW[:,i])[-1]))
                delta_w_n_prev =  np.resize(delta_w_n[:,i], (np.shape(dW[:,i])[-1]))
                delta_w_p_prev =  np.resize(delta_w_p[:,i], (np.shape(dW[:,i])[-1]))
            
            
            
            
            
            #train[i] = neuron.train
            #print(v_[i])
        return v, p_next, m, dW
    
    def multiple_neurons(self, stimuli, dt, time, n_neurons, stdp, p):
        ### the number of stimuli must be equal to the number of neurons

        steps = int(time/dt)
        v = np.zeros(steps)
        m = np.zeros(steps)
        p_next = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        delta_w_p =  np.zeros((n_neurons,steps))
        delta_w_n =  np.zeros((n_neurons,steps))
        dW =  np.zeros((n_neurons,steps))
        dW_prev = 0
        delta_w_n_prev = 0
        delta_w_p_prev = 0
        stimuli_conv = np.array(stimuli)
        stimuli = np.resize(stimuli_conv, (3*3,np.shape(stimuli)[-1]))
        W = np.zeros([3, 3])
        
        
        #weigths = np.random.randint(100, size=n_neurons)/50000
        for i in range(steps): 
            #for j in range(n_neurons):
                #weigthed_sum = weigths[j]*stimuli[j,i]
                #weigthed_sum = weight_ring(V=0.02*np.random.randint(1,11), stimulus=stimuli[j,i])
            stimuli_, W = conv_ring(input_=stimuli_conv[:,:,i], kernel_size=3, option=1)
            #print('debug stimuli', np.shape(stimuli_))            
            self.solve(stimuli_, dt, stdp=True, p=p)
            v[i] = self.V_p
            #print('debug2', np.shape(stimuli))
            if stdp:
                m[i] = self.post_trace
                p_next[i] = self.pre_trace
                delta_w_p[:,i], delta_w_n[:,i], dW[:,i] = self.stdp_function(m, n_neurons, stimuli, v, p, i, steps, dW_prev,delta_w_n_prev, delta_w_p_prev )
                dW_prev = np.resize(dW[:,i], (np.shape(dW[:,i])[-1]))
                delta_w_n_prev =  np.resize(delta_w_n[:,i], (np.shape(dW[:,i])[-1]))
                delta_w_p_prev =  np.resize(delta_w_p[:,i], (np.shape(dW[:,i])[-1]))
                

        return v, m, dW, p_next, W
    
    def stdp_function(self, m, n_neurons, stimuli, v, p, i, steps, dW_prev, delta_w_n_prev, delta_w_p_prev):
        delta_w_p =  np.zeros((n_neurons))
        delta_w_n =  np.zeros((n_neurons))
        dW =  np.zeros((n_neurons))
        for j in range(n_neurons):
            if stimuli[j,i] == 30:
                #print('debug')
                delta_w_n[j] = m[i]
                if i > 0:  dW[j] = dW_prev[j] + delta_w_n[j] # update weights
                #print('debug dW ', dW[j])
 
            ###################################################

            elif v[i] == 30 :
                temp = np.array(p)
                delta_w_p[j] = temp[j,i]
                if i > 0:  dW[j] = dW_prev[j] + delta_w_p[j] # update weights
            else:
                if i > 0: 
                    delta_w_n[j] = delta_w_n_prev[j]
                    delta_w_p[j] = delta_w_p_prev[j]
                    #dW[j] = dW_prev[j] # otherwise remains the same
                    #print('debug dW ', dW[j])
        
        return delta_w_p, delta_w_n, dW
        
        


       