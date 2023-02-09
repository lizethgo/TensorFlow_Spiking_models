#!/usr/bin/env python3
"""
File        : multiple_neurons.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : It generates the response of a single neuron, given n input 
              neurons (n_neurons).
"""
###############################################################################
## Libraries
import numpy as np
from LIF_solvers import LIF_RK, LIF_Euler
from functions import stimuli_gen, plot_multiple
###############################################################################

# program to compute the time
# of execution of any python code
import time as mytime
 
# we initialize the variable start
# to store the starting time of
# execution of program


### parameters    
time = 100
dt = 0.01
n_neurons = 3 # input neuroms
#LIF = LIF_RK() # Solve using Runge Kutta order 4th



# ### Generate n number of stimuli equal to n_neurons
I = [stimuli_gen(time, dt, 10, 40), 
      stimuli_gen(time, dt, 20, 70), 
      stimuli_gen(time, dt, 30, 60)]

start = mytime.time()

LIF_Euler(0.001).single_neuron(stimuli=I[0], dt=dt, time=time)

end = mytime.time()

# ### Generate n number of neurons equal to n_neurons
# start = mytime.time()
# v_in = [
# LIF_Euler(0.001).single_neuron(stimuli=I[0], dt=dt, time=time),
# LIF_Euler(0.001).single_neuron(stimuli=I[1], dt=dt, time=time),
# LIF_Euler(0.001).single_neuron(stimuli=I[2], dt=dt, time=time)]



# ### Generate weighted sum (random weights) of inputs
# v = LIF_Euler(C=0.001).multiple_neurons_ring(stimuli=v_in, dt=dt, time=time, n_neurons=n_neurons) #  If the number of sim points is reduced, C must be reduced too.
# end = mytime.time()

# vTime = np.arange(0, time, dt, dtype=None)

# ### plotting signals
# plot_multiple(time, dt, v, v_in, n_neurons=3)



 
# # difference of start and end variables
# # gives the time of execution of the
# # program in between
print("The time of execution of above program is :", end-start)