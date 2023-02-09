# -*- coding: utf-8 -*-
"""
##########################################################################################################
Author      : l.gonzalez.carabarin@tue.nl
Institution : Eindhoven University of Technology 
Description : Supervised training of SNN layers based on TF. This file contains layers and functions.
Date        : 17-06-2022

##########################################################################################################
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import math


def weight_ring(V):
    """
    Simple model based on the non-linear weighting of in/output.
    This model is somewhat normalized, takes [0,1] volts as input and calculates the corresponding weight [0,1]

    Takes V=[0, 1].
    Output: ring resonator TX == weight.
    (0 < Weight < 1)
    Written by Giuseppe Fortuna
    """
    # Waveguide parameters
    pi = tf.constant(math.pi, dtype=tf.float32)
    neff = tf.constant(3.6, dtype=tf.float32)
    wl = tf.constant(1.550183e-06, dtype=tf.float32)
    #wl = 1.550183e-06       # nominally OFF
    #wl = 1.549926e-06     # nominally ON

    # Ring parameters
    D=tf.constant(200e-6, dtype=tf.float32)
    #D = 200e-6          # diameter racetrack bends
    Ls = tf.constant(335.841e-6, dtype=tf.float32)
    #Ls = 335.841e-6     # length straight racetrack section
    
    C = pi*D + 2*Ls     # circumference ring
    r=tf.constant(0.1, dtype=tf.float32)
    #r = 0.1
    a = r

    # Phase shifter parameters
    #VLpi = 9e-3         # [Vmm] Voltage-length product for pi-shift (W.Yao)
    VLpi = tf.constant(9e-3, dtype=tf.float32)
    #Lp = 200e-6         # [Vmm] length phase shifter
    Lp = tf.constant(200e-6, dtype=tf.float32)
    
    Vpi = tf.math.round(VLpi/Lp)            # V-pi, 1mm phase shifter

    #V = tf.cast(V, tf.float32)
    #Vpi = tf.cast(Vpi, tf.float32)
    Vring = V*Vpi

    # E-field parameters
    dneff = (wl*Vring)/(2*Vpi)      # refractive index change
    dphi = 2*pi*dneff/wl            # additional phase from phase section
    phi = 2*pi*neff*C/wl + dphi     # total single-pass phase

    # Ring TX
    w = (a**2 + r**2 - 2*a*r*tf.math.cos(phi))/(1 + (a*r)**2 - 2*a*r*tf.math.cos(phi))
    #
    return w


"""
Solution of the LIF model based on Euler method
Notice that the method was adapated in such a way that execute parallely 
operations so the capabilities of TF can be leveraged.

"""

def f_v(v, I, R, C): 
    tm = (R * C)
    return (-v + R * I) / (tm)

def Euler(I, dt, V_p, R, C):
    dv = f_v(V_p, I, R, C) * dt    
    V_p.assign(tf.add(dv, V_p))
    return V_p

def solve(I, dt, R, C, V_thr, V_p, V_r, V_max): 

    ### if there is a spike, next step return to 0 
    V_p.assign(-tf.keras.activations.relu(-V_p, threshold=-27))
    V_p.assign(Euler(I, dt, V_p, R, C))
    ### If the membrane potential v is greater than the threshold, then generates a spike
    ### the use of sign and relu functions are a trick to execute this condition in parallel
    V_p.assign(14*(tf.math.sign(V_p-14)+1)-tf.keras.activations.relu(-V_p, threshold=-14))
    
    return V_p
    

def single_neuron(stimuli, dt, time,  V_thr, R, C, V_r, V_max, V_p):
    
    """
    This function calls Euler solver, and sequentially execute it across time.
    """

    steps = int(time/dt)
    V_p.assign(tf.zeros(V_p.shape, dtype=tf.float32))


    for i in range(200):
        
        V_ = solve(stimuli[:,:,:,:], dt, R, C, V_thr, V_p, V_r, V_max)
        if i == 0:
            V_temp = V_
        else:
            V_temp = (V_ + V_temp)


    return V_temp



class snn_layer_conv(Layer): 
    
    """
    Spiking Neural Network convolutional layer.
    
    """
    def __init__(self,  filters, channels, name):

        #self.name = 1
        self.filters = filters
        self.channels = channels

        super(snn_layer_conv, self).__init__(name=name)

    def build(self, input_shape): 
        self.k = self.add_weight(shape=[3,3, self.channels, self.filters],
                                 initializer=tf.keras.initializers.Constant(value=1),
                                 #initializer='he_uniform',
                                 #regularizer=regularizers.l2(0.0005),
                                 trainable=True, name='w', dtype='float32')
        
        self.V_thr = tf.constant(10.0, dtype=tf.float32) 
        self.R = tf.constant(3000.0, dtype=tf.float32)
        self.C = tf.constant(5, dtype=tf.float32)  # By reducing this value, the time constant tm is reduced
        self.V_r = tf.Variable(0.0, dtype=tf.float32,  trainable=False) # Resting potential
        self.V_max = tf.Variable(30.0, dtype=tf.float32,  trainable=False) #spike delta (V)
        self.v_p = np.zeros([input_shape[0], input_shape[1]-2,input_shape[2]-2, self.filters], dtype=np.float32)
        self.V_p = tf.Variable(self.v_p, dtype=tf.float32,  trainable=False)

    

        super(snn_layer_conv, self).build(input_shape) 
 
    def call(self, inp):

        time_ = 10
        dt = 0.01
        w = weight_ring(self.k)
        y = tf.nn.conv2d(inp, w, 1, padding = 'VALID')
        v = single_neuron(y, dt, time_, self.V_thr, self.R, self.C, self.V_r, self.V_max, self.V_p)
        v_ste = y # backwards
        
        return  tf.stop_gradient(v-v_ste)+v_ste







