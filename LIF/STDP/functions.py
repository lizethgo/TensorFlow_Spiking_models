#!/usr/bin/env python3
"""
File        : functions.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : Additional functions to generate multiple stimuli and 
              to visualize resulting inputs/outputs of LIF model.
             
"""
###############################################################################
## Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

###############################################################################


def stimuli_gen(time, dt, init, end):
    """
    Generates a single step stimulus for which init and end mark the beginning 
    and the end of the 'on' state.
    """
    steps = int(time/dt)
    I = [0.01 if (init/dt) < j < (end/dt)  
              else 0 
              for j in range(steps)]
    return I

def stimuli_gen_arr(time, dt, init, end):
    """
    Generates a single step stimulus for which init and end mark the beginning 
    and the end of the 'on' state.
    """
    steps = int(time/dt)
    I = [0.01 if (init/dt) < j < (end/dt)  
              else 0 
              for j in range(steps)]
    return I
    
    
def plot(time, dt,  v, I):
    """
    Plots the membrane potential v of a single LIF neuron together with its
    stimulus.
    v : output of the LIF neuron
    I : stimulus
    """
    t = np.arange(0, time, dt, dtype=None)
    fig, axs = plt.subplots(2)
    fig.suptitle('Single neuron')
    axs[0].plot(t, v)
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Voltage (v)")
    axs[1].plot(t, I, color='r')
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Current (I)")
    plt.show()
    
def plot_multiple(time, dt, v, v_in, n_neurons):
    """
    Plots the membrane potential v of a single LIF neuron as a result of multiple
    weighted input spikes.
    v        : output of the LIF neuron
    v_in     : list of the multiple input signals (n_neurons)
    n_neurons: number of input signals
    """
    t = np.arange(0, time, dt, dtype=None)
    fig, axs = plt.subplots(n_neurons+1)
    fig.suptitle('Multiple inputs')
    axs[0].plot(t, v)
    axs[0].set_ylabel("output (v)")
    axs[0].set_xticks([], minor=False) 
    
    for i in range(n_neurons):
        axs[i+1].plot(t, v_in[i], color=(np.random.randint(1,10)/10, 0, 0))
        if i == n_neurons-1:
            axs[i+1].set_xlabel("Time (ms)")
            axs[i+1].set_xticks([], minor=True)
        else:
            axs[i+1].set_xticks([], minor=False) 
        
        axs[i+1].set_ylabel("input {}".format(i+1))
    plt.show()
    
def plot_multiple_generic(time, dt, v_in, n, zoom, t_init, t_end, labels, title):
    """
    Plots the membrane potential v of a single LIF neuron as a result of multiple
    weighted input spikes.
    v        : output of the LIF neuron
    v_in     : list of the multiple input signals (n_neurons)
    n_neurons: number of input signals
    """
    t = np.arange(0, time, dt, dtype=None)
    fig, axs = plt.subplots(n)
    fig.suptitle(title)
    

    
    for i in range(n):
        #if zoom:
           # axs[i].set_xlim([int((time)*t_init),int((time)*t_end)])
           # axs[i].set_ylim([min(v_in[i][int((time)*t_init):int((time)*t_end)]), max(v_in[i][int((time)*t_init):int((time)*t_end)])])
        if zoom:    
            axs[i].plot(v_in[i][int((time/dt)*t_init):int((time/dt)*t_end)],  color=(np.random.randint(1,10)/10, np.random.randint(1,10)/10, np.random.randint(1,10)/10))
        else:
            axs[i].plot(t, v_in[i], color=(np.random.randint(1,10)/10, np.random.randint(1,10)/10, np.random.randint(1,10)/10))
        if i == n-1:
            axs[i].set_xlabel("Time (ms)")
            axs[i].set_xticks([], minor=True)
        else:
            axs[i].set_xticks([], minor=False) 
        
        #axs[i].set_ylabel("input {}".format(i+1))
        axs[i].set_ylabel(labels[i])


    plt.show()
def weight_ring(V, stimulus):
    """
    Simple model based on the non-linear weighting of in/output.
    This model is somewhat normalized, takes [0,1] volts as input and calculates the corresponding weight [0,1]

    Takes V=[0, 1].
    Output: ring resonator TX == weight.
    (0 < Weight < 1)
    Written by Giuseppe Fortuna
    """
    # Waveguide parameters
    pi = math.pi
    neff = 3.6
    wl = 1.550183e-06       # nominally OFF
    #wl = 1.549926e-06     # nominally ON

    # Ring parameters
    D = 200e-6          # diameter racetrack bends
    Ls = 335.841e-6     # length straight racetrack section
    C = pi*D + 2*Ls     # circumference ring
    r = 0.1
    a = r

    # Phase shifter parameters
    VLpi = 9e-3         # [Vmm] Voltage-length product for pi-shift (W.Yao)
    Lp = 200e-6         # [Vmm] length phase shifter
    Vpi = round(VLpi/Lp)            # V-pi, 1mm phase shifter
    Vring = V*Vpi

    # E-field parameters
    dneff = (wl*Vring)/(2*Vpi)      # refractive index change
    dphi = 2*pi*dneff/wl            # additional phase from phase section
    phi = 2*pi*neff*C/wl + dphi     # total single-pass phase

    # Ring TX
    w = (a**2 + r**2 - 2*a*r*np.cos(phi))/(1 + (a*r)**2 - 2*a*r*np.cos(phi))
    #
    return stimulus*w

    # stimulus=stimulus*w
    # return stimulus

def conv_ring(input_, kernel_size, option):
    
    #kernel = 0.0011*np.random.randint(1,100,size=(kernel_size, kernel_size,1))
    #kernel = 
    #kernel = w*np.ones([kernel_size, kernel_size,1])
    temp1 = 0
    if (option == 1):
        for i in range(kernel_size):
            for j in range(kernel_size):
                if j == 1:
                    temp = input_[i,j] *  weight_ring(V=0.01*np.random.randint(1,110), stimulus=1)
                    temp1 = temp + temp1 
                
        filter_ = temp1
        
    return filter_
    
    if (option == 2):
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == 1:
                    temp = input_[i,j] *  weight_ring(V=0.01*np.random.randint(1,110), stimulus=1)
                    temp1 = temp + temp1 
                
        filter_ = temp1
    return filter_
    
    if (option == 3):
        for i in range(kernel_size):
            for j in range(kernel_size):
                if j == i:
                    temp = input_[i,j] *  weight_ring(V=0.01*np.random.randint(1,110), stimulus=1)
                    temp1 = temp + temp1 
                
        filter_ = temp1
    return filter_

def temporal_decoding(data, size, time_points, print_):
    
    data_flatten = np.reshape(data,(size*size, np.shape(data)[-1]))
    data_flatten = np.transpose(data_flatten)
    rank_indexes = np.where(data_flatten>20)
    
    feature_map = np.zeros([1,size*size])
    a,ind = np.unique(rank_indexes[1][:],return_index=True)
    b =rank_indexes[0][ind]
    a = np.reshape(a,(1,np.shape(a)[0]))
    b = np.reshape(b,(1,np.shape(b)[0]))  ### N_points where a > 20
    t_1 = np.min(b)  ## first spike
    for i in range(np.shape(b)[1]):
        feature_map[0,a[0,i]] = np.abs(((b[0,i]-t_1)/time_points)-1)
    if print_:    
        feature_map = np.reshape(feature_map,(size,size))
        feature_map = 255*feature_map  ## ready to print
        plt.imshow(feature_map)
        plt.show()

    
    return feature_map