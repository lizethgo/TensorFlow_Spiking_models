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

########################################################################################################
########################################################################################################
#%% Load data  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

########################################################################################################
########################################################################################################


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
########################################################################################################
########################################################################################################
    
### parameters    
time_ = 10
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
    #print("debug", tf.shape(dv))
    #print("Euler debug", dv, V_p)
    #print(V_p)
    
    V_p.assign(tf.add(dv, V_p))
    return V_p

def solve(I, dt, R, C, V_thr, V_p, V_r, V_max): 

    ### If the membrane potential v is greater than the threshold
    ### returns to the resting potential V_r
    #print('I', np.shape(I))

    
    

    #V_p.assign(tf.keras.activations.relu(30, threshold=15)) ### 
    V_p.assign(-tf.keras.activations.relu(-V_p, threshold=-27))
    #print(V_p)
    V_p.assign(Euler(I, dt, V_p, R, C))
    #print(V_p)
    
    
    #return V_p.assign_add(V_p,tf.keras.activations.relu(V_p, threshold=15))
    return V_p.assign(14*(tf.math.sign(V_p-14)+1)-tf.keras.activations.relu(-V_p, threshold=-14))
    #print(V_p)
    




def single_neuron(stimuli, dt, time, R, C, V_thr, V_r, V_max):
    steps = int(time/dt)
    v_plot = np.zeros([steps, np.shape(stimuli)[0]])
    v = tf.zeros([steps, np.shape(stimuli)[0]], dtype=np.float32)
    v = tf.Variable(v)
    #v[0].assign(0)
    v_p = np.zeros([np.shape(stimuli)[0]], dtype=np.float32)
    V_p = tf.Variable(v_p, dtype=tf.float32)
    #print(V_p)

    for i in range(steps):
        V_ = solve(stimuli[:,i], dt, R, C, V_thr, V_p, V_r, V_max)
        #print(V_.numpy()[0])
        #print(np.shape(V_))

        #plt.plot(V_.numpy()[0])
        #plt.show()
        v[i,:].assign(V_p)
        v_plot[i,:] = v[i,:].numpy()
    plt.plot(v_plot[:,0])
    plt.show()
    plt.plot(v_plot[:,1])
    plt.show()
    plt.plot(v_plot[:,2])
    plt.show()
        
    return V_






class snn_layer_conv(Layer): 
    
    """
    Sparse convolutional layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify k matrix (kernels)
    """
    def __init__(self, name=None):

        #self.name = 1
        super(snn_layer_conv, self).__init__(name=name)

    def build(self, input_shape): 

         super(snn_layer_conv, self).build(input_shape) 
 
    def call(self, inp):

        time_ = 10
        dt = 0.01
        

        V_thr = tf.constant(10.0, dtype=tf.float32) 
        R = tf.constant(3000.0, dtype=tf.float32)
        C = tf.constant(10, dtype=tf.float32)  # By reducing this value, the time constant tm is reduced
        V_r = tf.Variable(0.0, dtype=tf.float32) # Resting potential
        V_max = tf.Variable(30.0, dtype=tf.float32) #spike delta (V)
        k = tf.ones(shape=[3,3,1,1], dtype='float32')
        
        plt.imshow(inp[0,:,:,0])
        y = tf.nn.conv2d(inp, k, 1, padding = 'VALID')
        plt.imshow(y[0,:,:,0])
        #print("deb", y)
        v = single_neuron(y, dt, time_, R, C, V_thr, V_r, V_max)
        #print(np.shape(v))
        
        return v
#

I = [stimuli_gen(time_, dt, 1, 4), 
stimuli_gen(time_, dt, 2, 7), 
stimuli_gen(time_, dt, 3, 6)]
I_test = tf.convert_to_tensor(I, dtype=tf.float32)




V_thr = tf.constant(10.0, dtype=tf.float32) 
R = tf.constant(3000.0, dtype=tf.float32)
C = tf.constant(10, dtype=tf.float32)  # By reducing this value, the time constant tm is reduced
V_r = tf.Variable(0.0, dtype=tf.float32) # Resting potential
V_max = tf.Variable(30.0, dtype=tf.float32) #spike delta (V)
k = tf.ones(shape=[3,3,1,1], dtype='float32')

#plt.imshow(inp[0,:,:,0])
#y = tf.nn.conv2d(inp, k, 1, padding = 'VALID')
#plt.imshow(y[0,:,:,0])
#print("deb", y)
v = single_neuron(I_test, dt, time_, R, C, V_thr, V_r, V_max)

# import time as mytime
# start = mytime.time()        
# snn_layer_ = snn_layer_conv()
# inp = tf.reshape(x_train[0], shape=[1,tf.shape(x_train)[1], tf.shape(x_train)[2], tf.shape(x_train)[-1]])
# v = snn_layer_(inp)
# end = mytime.time()
# print("The time of execution of above program is :", end-start)
# #plt.plot(v)





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
   
        