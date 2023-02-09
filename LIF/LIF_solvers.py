#!/usr/bin/env python3
"""
File        : LIF_solvers.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : This file contains the classes:
              1. LIF_RK tha solves the LIF model equation based on the Runge Kutta 4th order
              2. LIF_Euler that solves the LIF model equation based on the Euler Method
             
References: https://www.researchgate.net/publication/322568485_Spike_Neural_Models_Part_II_Abstract_Neural_Models


    
"""
###############################################################################
## Libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from functions import weight_ring
###############################################################################

    
  
class LIF_RK:
    """
    v(t)/dt = (-v(t) + RI(t))/tm
    where tm = R*C
    
    if == v(t)>vth  -> v(t) = V_max and v(t+1) = V_r
    
    """ 
    def __init__(self, C,**kwargs):
        self.V_thr = 15 
        self.R = 3000 
        self.C = C  # By reducing this value, the time constant tm is reduced 0.001
        self.V_r = 0 # Resting potential
        self.V_max = 30 # Maximum values of an spike 
        self.V_p = -1 # initial value
        self.train = 0
        
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

    def solve(self, I, dt): 

        ### If the membrane potential v is greater than the threshold
        ### returns to the resting potential V_r
        if (self.V_p >= self.V_thr):
            self.V_p = self.V_r
            self.train = 0  ## to exclusuvely observe the spikes 
        else: 
            self.train =0 ## to exclusuvely observe the spikes 
            self.RungeKutta4(I, dt)
        ### If the new value of the membrane potential v is greater than the threshold
        ### it generates an spike of mag 40
        if self.V_p >= self.V_thr: 
            self.V_p = self.V_max
            self.train = 30



    
    def single_neuron(self, stimuli, dt, time):
        steps = int(time/dt)
        v = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        for i in range(steps): 
            self.solve(stimuli[i], dt)
            v[i] = self.V_p
            #train[i] = neuron.train
            #print(v_[i])
        return v
    def multiple_neurons(self, stimuli, dt, time, n_neurons):
        ### the number of stimuli must be equal to the number of neurons
        steps = int(time/dt)
        v = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        weigths = np.random.randint(100, size=n_neurons)/50000
        for i in range(steps): 
            for j in range(n_neurons):
                weigthed_sum = weigths[j]*stimuli[j][i]
            
            self.solve(weigthed_sum, dt)
            v[i] = self.V_p
        return v
    
    def multiple_neurons_ring(self, stimuli, dt, time, n_neurons):
        ### the number of stimuli must be equal to the number of neurons
        steps = int(time/dt)
        v = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        weigths = np.random.randint(100, size=n_neurons)/50000
        for i in range(steps): 
            for j in range(n_neurons):
                
                weigthed_sum = weight_ring(V=0.02*np.random.randint(1,11), stimulus=stimuli[j][i])
            
            self.solve(weigthed_sum, dt)
            v[i] = self.V_p
        return v

        
        
        
        
    

        
        
    

class LIF_Euler:
    """
    v(t)/dt = (-v(t) + RI(t))/tm
    where tm = R*C
    
    if == v(t)>V_th  -> v(t) = V_max and v(t+1) = V_r
    
    """ 
    def __init__(self, C,**kwargs):
        self.V_thr = 15 
        self.R = 3000 
        self.C = C  # By reducing this value, the time constant tm is reduced 0.001
        self.V_r = 0 # Resting potential
        self.V_max = 30 # Maximum values of an spike 
        self.V_p = -1 # initial value
        self.train = 0
        
        
    def Euler(self, I, dt):
        dv = self.f_v(self.V_p, I) * dt 
        self.V_p += dv

    def solve(self, I, dt): 

        ### If the membrane potential v is greater than the threshold
        ### returns to the resting potential V_r
        if (self.V_p >= self.V_thr):
            self.V_p = self.V_r
            self.train = 0  ## to exclusuvely observe the spikes 
        else: 
            self.train =0 ## to exclusuvely observe the spikes 
            self.Euler(I, dt)
        ### If the new value of the membrane potential v is greater than the threshold
        ### it generates an spike of mag 40
        if self.V_p >= self.V_thr: 
            self.V_p = self.V_max
            self.train = 30


    def f_v(self, v, I): 
        tm = (self.R * self.C)
        return (-v + self.R * I) / tm
    
    def single_neuron(self, stimuli, dt, time):
        steps = int(time/dt)
        v = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        for i in range(steps): 
            self.solve(stimuli[i], dt)
            v[i] = self.V_p

        return v
    def multiple_neurons(self, stimuli, dt, time, n_neurons):
        ### the number of stimuli must be equal to the number of neurons
        steps = int(time/dt)
        v = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        weigths = np.random.randint(100, size=n_neurons)/50000
        for i in range(steps): 
            for j in range(n_neurons):
                weigthed_sum = weigths[j]*stimuli[j][i]
            
            self.solve(weigthed_sum, dt)
            v[i] = self.V_p
        return v
    
    def multiple_neurons_ring(self, stimuli, dt, time, n_neurons):
        ### the number of stimuli must be equal to the number of neurons
        steps = int(time/dt)
        v = np.zeros(steps)
        v[0] = self.V_p
        v = np.zeros(steps)
        weigths = np.random.randint(100, size=n_neurons)/50000
        for i in range(steps): 
            for j in range(n_neurons):
                
                weigthed_sum = weight_ring(V=0.02*np.random.randint(1,11), stimulus=stimuli[j][i])
            
            self.solve(weigthed_sum, dt)
            v[i] = self.V_p
        return v


        
        
        
        
    
