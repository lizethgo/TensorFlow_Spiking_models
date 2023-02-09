#!/usr/bin/env python3
"""
File        : single_neuron.py
Author      : Lizeth Gonzalez Carabarin
Institution : Eindhoven University of Technology
Description : It generates the response of a single neurons given a single input (step)

"""
###############################################################################
## Libraries

from LIF_solvers import LIF_RK
from functions import stimuli_gen, plot
###############################################################################

##### single neuron ###########################################################
## parameters
time = 100 # Total simulation time
dt = 0.01
I = stimuli_gen(time=time, dt=0.01, init=30, end=70) # Generates the input stimuli
LIF = LIF_RK(C=0.001) # Solve using Runge Kutta order 4th. If the number of sim points is reduced, C must be reduced too.
v = LIF.single_neuron(stimuli=I, dt=dt, time=time) # Single neuron
plot(time, dt, v, I) # plot resulting membrane potential and input stimuli











