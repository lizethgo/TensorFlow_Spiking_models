#!/usr/bin/env python3
"""
File        : LIF_STDP.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : It generates the response of a single neuron, given n input 
              neurons (n_neurons). Aditionally, shows graphically the STDP rule over the weights.
"""
###############################################################################
## Libraries
import numpy as np
from LIF_solver_STDP import LIF_RK
from functions import stimuli_gen, plot_multiple, plot_multiple_generic
###############################################################################

### parameters    
# time = 100
# dt = 0.01
# n_neurons = 3 # input neuroms
# LIF = LIF_RK() # Solve using Runge Kutta order 4th

# ### Generate n number of stimuli equal to n_neurons
# I = [stimuli_gen(time, dt, 10, 40), 
#      stimuli_gen(time, dt, 20, 70), 
#      stimuli_gen(time, dt, 30, 60)]

time = 5
dt = 0.01
n_neurons = 3 # input neuroms
LIF = LIF_RK(0.00005) # Solve using Runge Kutta order 4th.  If the number of sim points is reduced, C must be reduced too.

I = [stimuli_gen(time, dt, 0.5, 2), 
     stimuli_gen(time, dt, 1, 3.5), 
     stimuli_gen(time, dt, 1.5, 3),
     stimuli_gen(time, dt, 2, 4.5),
     stimuli_gen(time, dt, 4, 4.5)]

p=0
### Generate n number of neurons equal to n_neurons
v_in = [
LIF.single_neuron(stimuli=I[0], dt=dt, time=time, stdp=False, p=p)[0],
LIF.single_neuron(stimuli=I[1], dt=dt, time=time, stdp=False, p=p)[0],
LIF.single_neuron(stimuli=I[2], dt=dt, time=time, stdp=False, p=p)[0]]

### Generate n number of neurons equal to n_neurons
p_in = [
LIF.single_neuron(stimuli=I[0], dt=dt, time=time, stdp=False, p=p)[1],
LIF.single_neuron(stimuli=I[1], dt=dt, time=time, stdp=False, p=p)[1],
LIF.single_neuron(stimuli=I[2], dt=dt, time=time, stdp=False, p=p)[1]]

#delta_w_p = [
#LIF.single_neuron(stimuli=I[0], dt=dt, time=time)[2],
#LIF.single_neuron(stimuli=I[1], dt=dt, time=time)[2],
#LIF.single_neuron(stimuli=I[2], dt=dt, time=time)[2]]
#
#delta_w_n = [
#LIF.single_neuron(stimuli=I[0], dt=dt, time=time)[3],
#LIF.single_neuron(stimuli=I[1], dt=dt, time=time)[3],
#LIF.single_neuron(stimuli=I[2], dt=dt, time=time)[3]]

### Generate weighted sum (random weights) of inputs
v, m, delta_w_p, delta_w_n, dW, p_next = LIF.multiple_neurons(stimuli=v_in, dt=dt, time=time, n_neurons=n_neurons, p=p_in, stdp=True)    
#time = np.arange(0, time, dt, dtype=None)

### plotting signals
plot_multiple(time, dt, v, v_in, n_neurons=3)
plot_multiple(time, dt, m, p_in, n_neurons=3)
#plot_multiple(time, dt, delta_w_n[0], delta_w_p, n_neurons=3)

title = ''
labels = ['delta_w_1', 'delta_w_1', 'delta_w_1' ]
data = [delta_w_p[0] + delta_w_n[0], delta_w_p[1] + delta_w_n[1], delta_w_p[2] + delta_w_n[2]]
plot_multiple_generic(time, dt, dW, n=3, zoom=False,t_init=0.3, t_end=0.4, title = title, labels = labels)

title = ''
labels = ['v', 'v_in_1', 'delta_w_p' ]
data = [v, v_in[0], delta_w_p[0]]
plot_multiple_generic(time, dt, data, n=3, zoom=True, t_init=0.3, t_end=0.4, title = title, labels = labels)

title = ''
labels = ['m', 'p_in_1', 'delta_w_p' ]
data = [m, p_in[0], delta_w_p[0]]
plot_multiple_generic(time, dt, data, n=3, zoom=True, t_init=0.3, t_end=0.4, title = title, labels = labels)

title = ''
labels = ['m', 'p_in_1', 'delta_w_n' ]
data = [m, p_in[0], delta_w_n[0]]
plot_multiple_generic(time, dt, data, n=3, zoom=True, t_init=0.3, t_end=0.4, title = title, labels = labels)

title = ''
labels = ['m', 'p_in_1', 'dW' ]
data = [m, p_in[0], dW[0]]
plot_multiple_generic(time, dt, data, n=3, zoom=True, t_init=0.3, t_end=0.4, title = title, labels = labels)


#### Additional example
from LIF_solver_STDP import LIF_RK
from functions import stimuli_gen, plot_multiple, plot_multiple_generic
I_ = [stimuli_gen(time, dt, 0.5, 2)]
I_ - np.array(I_)
v1, p1, m1, dW1 = LIF.single_neuron(stimuli=I_, dt=dt, time=time, stdp=False, p=p)
v2, p2, m2, dW2 = LIF.single_neuron(stimuli=v1, dt=dt, time=time, stdp=True, p=p1)
title = ''
labels = ['v1', 'v2', 'p1', 'm', 'dW2' ]
data = [v1, v2, p1, m2, dW2[0]]
plot_multiple_generic(time, dt, data, n=5, zoom=True, t_init=0.3, t_end=0.4, title = title, labels = labels)


