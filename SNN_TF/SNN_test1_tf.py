# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
#import pylab as pl

#sess = tf.InteractiveSession()

## LIF parameters
Rm = 1                # resistance (kOhm)
Cm = 10               # capacity (uF)
tau_m = Rm*Cm         # time constant (msec)
tau_ref = 4           # refractory period (msec)
Vth = 1               # spike threshold
V_spike = 0.5         # spike delta (V)
V_rest = 0.0          # resting potential

## Define Computation Graph. TF2 does not support placeholders
#I = tf.compat.v1.placeholder('float32', shape=())  # Input current
#dt = tf.compat.v1.placeholder('float32', shape=()) # simulation time step (msec)

def stimuli_gen(time, dt, init, end):
    """
    Generates a single step stimulus for which init and end mark the beginning 
    and the end of the 'on' state.
    """
    steps = int(time/dt)
    I = [150.0 if (init/dt) < j < (end/dt)  
              else 0 
              for j in range(steps)]
    return I


### parameters    
time_ = 100
dt = 0.01




### Generate n number of stimuli equal to n_neurons
I_ = [stimuli_gen(time_, dt, 10, 40)]

#### Convert to tensors
I_test = tf.convert_to_tensor(I_)
#dt = tf.convert_to_tensor(dt_)
time = tf.convert_to_tensor(time_)


Vm = tf.Variable(0.0, name='Vm')         # potential (V) trace over time
t_rest = tf.Variable(0.0, name='t_rest') # 

def euler(I):
    dV = (-Vm + I*Rm) / tau_m 
    #print(tf.shape( dV))
    test = dV * dt
    #print(tf.shape(Vm))
    a = Vm.assign_add(dV * dt)
    b = t_rest
    return a

def spiking(I):
    a = Vm.assign_add(V_spike)
    b = t_rest.assign(tau_ref)
    return a

def resting(I):
    a = Vm.assign(V_rest) # TODO: could be done only ones
    b = t_rest.assign_sub(dt)
    return a




#tf.global_variables_initializer().run()
#times = pl.arange(0, T, dt_val)
results = []

for i in range(10000):
    I = I_test[0,i]
    #res = sess.run(step, feed_dict={I: I_val, dt: dt_val})
    
    if tf.greater (t_rest, 0): 
        res = resting(I)
    if tf.greater (Vm, Vth):  
        res = spiking(I)
    else:
        res = euler(I)
    
    
    #res = step
    results.append(res.numpy())
    #writer.add_summary(summary, i)



plt.plot(results)
plt.title('Leaky Integrate-and-Fire Example')
plt.ylabel('Membrane Potential (V)')
plt.xlabel('Time (msec)')
plt.ylim([0,2])
plt.show()

