# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:32:11 2022

@author: 20195088
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import numpy as np
from functions_tf import weight_ring, plot_multiple_generic, plot_multiple
from tensorflow.keras.layers import Layer

# program to compute the time
# of execution of any python code
import time as mytime
 
# we initialize the variable start
# to store the starting time of
# execution of program


def stimuli_gen(time, dt, init, end):
    """
    Generates a single step stimulus for which init and end mark the beginning 
    and the end of the 'on' state.
    """
    steps = int(time/dt)
    I = [150.0 if (init/dt) < j < (end/dt)  
              else 0.0 
              for j in range(steps)]
    return I

### parameters    
time_ = 100
dt = 0.01

### Generate n number of stimuli equal to n_neurons
I_ = [stimuli_gen(time_, dt, 1, 4)]
#### Convert to tensors
I_test = tf.convert_to_tensor(I_)

class LIF_euler:
    
    def __init__(self, dt, time, **kwargs):

        self.V_thr = tf.constant(15.0, dtype=tf.float32) 
        self.R = tf.constant(3000.0, dtype=tf.float32)
        self.C = tf.constant(0.001, dtype=tf.float32)  # By reducing this value, the time constant tm is reduced
        self.V_r = tf.Variable(0.0, dtype=tf.float32) # Resting potential
        self.V_max = tf.Variable(30.0, dtype=tf.float32) #spike delta (V)
        self.tau_m = tf.constant(self.R*self.C, dtype=tf.float32)
        self.tau_ref = tf.constant(5, dtype=tf.float32)
        self.V_p = tf.Variable(-1.0, name='t_rest', dtype=tf.float32)  # initial value
        #self.V_p = -1 # initial value
        #self.train = 0
        self.Vm = tf.Variable(0.0, name='Vm', dtype=tf.float32)         # potential (V) trace over time
        self.t_rest = tf.Variable(0.0, name='t_rest', dtype=tf.float32) #
        self.dt = dt
        self.time = time
        self.steps = int(self.time/self.dt)

            

        
    def Euler(self, I, dt):
        dv = self.f_v(self.V_p, I) * dt 
        
        self.V_p.assign_add(dv)

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
        v_plot = np.zeros(steps)
        v = tf.zeros(steps, dtype=np.float32)
        v = tf.Variable(v)
        v[0].assign(self.V_p)

        for i in range(steps): 
            self.solve(stimuli[i], dt)
            v[i].assign(self.V_p)
            
        return v    
    
    def solve_multiple_neurons(self, I_test, n_neurons):
        results = np.zeros([self.steps])
        
        weigths = tf.random.uniform(shape=[n_neurons,1])/500
        I_weighted=tf.reduce_sum(I_test*weigths, axis=0)

        for i in range(self.steps):

            if tf.greater (self.t_rest, 0): 
                res = self.resting(I_weighted[i])
            elif tf.greater (self.Vm, self.V_thr):  
                res = self.spiking(I_weighted[i])
            else:
                res = self.euler(I_weighted[i])
        
            results[i]=res.numpy()
        return results
    
    def solve_multiple_neurons_ring(self, I_test, n_neurons):
        results = np.zeros([self.steps])
        

        weigths = weight_ring(V=0.1*tf.random.uniform(shape=[n_neurons,1]))
        #print(weigths)
        I_weighted=tf.reduce_sum(I_test*weigths, axis=0)
        #print(tf.shape(I_weighted))

        for i in range(self.steps):

            if tf.greater (self.t_rest, 0): 
                res = self.resting(I_weighted[i])
            elif tf.greater (self.Vm, self.V_thr):  
                res = self.spiking(I_weighted[i])
            else:
                res = self.euler(I_weighted[i])
        
            results[i]=res.numpy()
                
        return results
    

# LIF = LIF_euler(0.01, 100.0)

# ### Generate n number of stimuli equal to n_neurons
# ### Generate n number of stimuli equal to n_neurons
# I = [stimuli_gen(time_, dt, 10, 40), 
#       stimuli_gen(time_, dt, 20, 70), 
#       stimuli_gen(time_, dt, 30, 60)]


# I_test = tf.convert_to_tensor(I, dtype=tf.float32)


# start = mytime.time()
# LIF.single_neuron(I_test[0], 0.01, 100.0)
# end = mytime.time()  


def f_v(v, I, R, C): 
    tm = (R * C)
    return (-v + R * I) / tm

def Euler(I, dt, V_p, R, C):
    dv = f_v(V_p, I, R, C) * dt 
    #print("Euler debug", dv, V_p)
    #print(V_p)
    
    V_p.assign_add(dv)
    return V_p

def solve(I, dt, R, C, V_thr, V_p, V_r, V_max): 

    ### If the membrane potential v is greater than the threshold
    ### returns to the resting potential V_r
    #print('I', np.shape(I))
    #print(V_p)
    if (V_p >=V_thr):
        #print('RESTING')
        V_p.assign(V_r)
        train = 0  ## to exclusuvely observe the spikes 
    else: 
        train =0 ## to exclusuvely observe the spikes 
        V_p = Euler(I, dt, V_p, R, C)
    ### If the new value of the membrane potential v is greater than the threshold
    ### it generates an spike of mag 40
    if V_p >= V_thr: 
        V_p.assign(V_max)
        train = 30
    return V_p




def single_neuron(stimuli, dt, time, R, C, V_thr, V_r, V_max):
    steps = int(time/dt)
    v_plot = np.zeros(steps)
    v = tf.zeros(steps, dtype=np.float32)
    v = tf.Variable(v)
    v[0].assign(0)
    V_p = tf.Variable(0, dtype=tf.float32)

    for i in range(steps):
        V_p = solve(stimuli[i], dt, R, C, V_thr, V_p, V_r, V_max)
        v[i].assign(V_p)
        v_plot[i] = v[i].numpy()
        
    return v_plot

def multiple_neuron(stimuli, dt, time, R, C, V_thr, V_r, V_max, n_neurons):
    steps = int(time/dt)
    v_plot = np.zeros(steps)
    v = tf.zeros(steps, dtype=np.float32)
    v = tf.Variable(v)
    v[0].assign(0)
    V_p = tf.Variable(0, dtype=tf.float32)
    
    weigths = tf.random.uniform(shape=[n_neurons,1])
    I_weighted=tf.reduce_sum(stimuli*weigths, axis=0)

    for i in range(steps):
        V_p = solve(I_weighted[i], dt, R, C, V_thr, V_p, V_r, V_max)
        v[i].assign(V_p)
        v_plot[i] = v[i].numpy()
        
    return v_plot

class snn_layer(Layer): 
    
    """
    Sparse convolutional layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify k matrix (kernels)
    """
    def __init__(self, name=None):

        #self.name = 1
        super(snn_layer, self).__init__(name=name)

    def build(self, input_shape): 

         super(snn_layer, self).build(input_shape) 
 
    def call(self, inp):

        time_ = 1
        dt = 0.01
        
        I = [stimuli_gen(time_, dt, 0.1, 0.4), 
        stimuli_gen(time_, dt, 0.2, 0.7), 
        stimuli_gen(time_, dt, 0.3, 0.6)]
        
        I_test = tf.convert_to_tensor(I, dtype=tf.float32)
        
        
        V_thr = tf.constant(15.0, dtype=tf.float32) 
        R = tf.constant(3000.0, dtype=tf.float32)
        C = tf.constant(0.05, dtype=tf.float32)  # By reducing this value, the time constant tm is reduced
        V_r = tf.Variable(0.0, dtype=tf.float32) # Resting potential
        V_max = tf.Variable(30.0, dtype=tf.float32) #spike delta (V)
        


        v = single_neuron(I_test[0], dt, time_, R, C, V_thr, V_r, V_max)
        #print(np.shape(v))
        
        return v


class snn_layer_fc(Layer): 
    
    """
    Sparse convolutional layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify k matrix (kernels)
    """
    def __init__(self, name=None):

        #self.name = 1
        super(snn_layer_fc, self).__init__(name=name)

    def build(self, input_shape): 

         super(snn_layer_fc, self).build(input_shape) 
 
    def call(self, inp):

        time_ = 10
        dt = 0.01
        
        I = [stimuli_gen(time_, dt, 1, 4), 
        stimuli_gen(time_, dt, 2, 7), 
        stimuli_gen(time_, dt, 3, 6)]
        
        I_test = tf.convert_to_tensor(I, dtype=tf.float32)
        
        
        V_thr = tf.constant(10.0, dtype=tf.float32) 
        R = tf.constant(3000.0, dtype=tf.float32)
        C = tf.constant(10, dtype=tf.float32)  # By reducing this value, the time constant tm is reduced
        V_r = tf.Variable(0.0, dtype=tf.float32) # Resting potential
        V_max = tf.Variable(30.0, dtype=tf.float32) #spike delta (V)
        


        v = multiple_neuron(I_test, dt, time_, R, C, V_thr, V_r, V_max, 3)
        #print(np.shape(v))
        
        return v
#

snn_layer_ = snn_layer_fc()
v = snn_layer_(0)




# v_in = [
# LIF.solve_single_neuron(I_test[0]),
# LIF.solve_single_neuron(I_test[1]),
# LIF.solve_single_neuron(I_test[2])]

# v_in_tf = tf.convert_to_tensor(v_in, dtype=tf.float32)

# results = LIF.solve_multiple_neurons_ring(v_in_tf, 3)
      

# plot_multiple(time_, dt, results, I, n_neurons=3)



# #results = LIF.solve_multiple_neurons_ring(I_test, 3)
# #plot_multiple(time_, dt, results, I, n_neurons=3)



 
# # difference of start and end variables
# # gives the time of execution of the
# # program in between
#print("The time of execution of above program is :", end-start)
   
        