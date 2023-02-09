#!/usr/bin/env python3
"""
File        : multiple_neurons.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : It generates the response of a single neuron, given n input 
              neurons (n_neurons) connected with the ring resonator model
"""
###############################################################################
## Libraries
import numpy as np
from LIF_solvers import LIF_RK
from functions import stimuli_gen, plot_multiple
###############################################################################

### parameters    
time = 5
dt = 0.01
n_neurons = 3 # input neuroms
LIF = LIF_RK(0.00005) # Solve using Runge Kutta order 4th.  If the number of sim points is reduced, C must be reduced too.


### Generate n number of stimuli equal to n_neurons
I = [stimuli_gen(time, dt, 0.5, 2), 
     stimuli_gen(time, dt, 1, 3.5), 
     stimuli_gen(time, dt, 1.5, 3),
     stimuli_gen(time, dt, 2, 4.5),
     stimuli_gen(time, dt, 4, 4.5)]

### Generate n number of neurons equal to n_neurons
v_in = [
LIF.single_neuron(stimuli=I[0], dt=dt, time=time),
LIF.single_neuron(stimuli=I[1], dt=dt, time=time),
LIF.single_neuron(stimuli=I[2], dt=dt, time=time),
LIF.single_neuron(stimuli=I[3], dt=dt, time=time),
LIF.single_neuron(stimuli=I[4], dt=dt, time=time)]

### Generate weighted sum (random weights) of inputs
v = LIF.multiple_neurons_ring(stimuli=v_in, dt=dt, time=time, n_neurons=n_neurons)
vTime = np.arange(0, time, dt, dtype=None)

### plotting signals
plot_multiple(time, dt, v, v_in, n_neurons=5)