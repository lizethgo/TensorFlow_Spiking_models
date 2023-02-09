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
from tensorflow.keras import backend as K
 
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

x_val = x_train[-5000:,:,:,:]
x_train = x_train[:-5000,:,:,:]

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

y_val = y_train[-5000:,:]
y_train = y_train[:-5000,:]

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
    return (-v + R * I) / (tm)

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

    
    

    #V_p.assign(-tf.keras.activations.relu(-V_p, -29)) ### 
    #print(V_p)
    #V_p.assign(Euler(I, dt, V_p, R, C))
    #print(V_p)
    
    
    #return V_p.assign_add(V_p,tf.keras.activations.relu(V_p, threshold=15))
    #print(V_p)
    V_p.assign(-tf.keras.activations.relu(-V_p, threshold=-27))
    #print(V_p)
    V_p.assign(Euler(I, dt, V_p, R, C))
    #print(V_p)
    
    
    #return V_p.assign_add(V_p,tf.keras.activations.relu(V_p, threshold=15))
    return V_p.assign(14*(tf.math.sign(V_p-14)+1)-tf.keras.activations.relu(-V_p, threshold=-14))
    




def single_neuron(stimuli, dt, time,  V_thr, R, C, V_r, V_max, V_p):

   
    
    steps = int(time/dt)
    #v_plot = np.zeros([steps, np.shape(stimuli)[1],np.shape(stimuli)[2]])
    #v = tf.zeros([steps, np.shape(stimuli)[1],np.shape(stimuli)[2]], dtype=np.float32)
    #v = tf.Variable(v)
    #v[0].assign(0)
    #v_p = np.zeros([np.shape(stimuli)[0], np.shape(stimuli)[1],np.shape(stimuli)[2]], dtype=np.float32)
    #V_p = tf.Variable(v_p, dtype=tf.float32)
    #print(V_p)
    V_p.assign(tf.zeros(V_p.shape, dtype=tf.float32))

    for i in range(1000):
        
        V_ = solve(stimuli[:,:,:,0], dt, R, C, V_thr, V_p, V_r, V_max)
        if i == 0:
            V_temp = V_
        else:
            V_temp = (V_ + V_temp)/1000
        #print(np.shape(V_))
        #print(np.shape(K.eval(V_)))
        #print(V_)
        #plt.imshow(V_[0,:,:,0])
        #plt.show()
        #v[i,:,:].assign(V_p)
        #v_plot[i,:,:] = v[i,:,:].numpy()
        
    return (V_temp)






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
        self.k = self.add_weight(shape=[3,3, 1, 1],
                                 initializer=tf.keras.initializers.Constant(value=1),
                                 #regularizer=regularizers.l2(0.0005),
                                 trainable=True, name='w', dtype='float32')
        
        self.V_thr = tf.constant(10.0, dtype=tf.float32) 
        self.R = tf.constant(3000.0, dtype=tf.float32)
        self.C = tf.constant(10, dtype=tf.float32)  # By reducing this value, the time constant tm is reduced
        self.V_r = tf.Variable(0.0, dtype=tf.float32,  trainable=False) # Resting potential
        self.V_max = tf.Variable(30.0, dtype=tf.float32,  trainable=False) #spike delta (V)
        print("input shape", input_shape)
        self.v_p = np.zeros([input_shape[0], input_shape[1]-2,input_shape[2]-2], dtype=np.float32)
        self.V_p = tf.Variable(self.v_p, dtype=tf.float32,  trainable=False)

        super(snn_layer_conv, self).build(input_shape) 
 
    def call(self, inp):

        time_ = 10
        dt = 0.01
        print("inp", inp.shape[0], inp.shape[1],inp.shape[2] )
        #k = tf.ones(shape=[3,3,1,1], dtype='float32')
        


        
        #plt.imshow(inp[0,:,:,0])
        y = tf.nn.conv2d(inp, self.k, 1, padding = 'VALID')
        #print(tf.shape(y))
        #plt.imshow(y.numpy()[0,:,:,0])
        #print("deb", y)
        
        v = single_neuron(y, dt, time_, self.V_thr, self.R, self.C, self.V_r, self.V_max, self.V_p)
        v = tf.reshape(v, shape=(v.shape[0], v.shape[1], v.shape[2], 1))
        v_ste = y # backwards
       
        #tf.stop_gradient(hardSamples - softSamples) + softSamples
        #print(np.shape(v))
        
            
        
        return  tf.stop_gradient(v-v_ste)+v_ste
        #return  y
#


#import time as mytime
#start = mytime.time()        
#snn_layer_ = snn_layer_conv()
#inp = tf.reshape(x_train[0], shape=[1,tf.shape(x_train)[1], tf.shape(x_train)[2], tf.shape(x_train)[-1]])
#v = snn_layer_(inp)
#end = mytime.time()
#print("The time of execution of above program is :", end-start)
#plt.plot(v)



class sample_vis(tf.keras.callbacks.Callback):
    def __init__(self, data, **kwargs):
        self.data = data
    def on_epoch_begin(self, epoch, logs=None):
        #D1 = self.model.layers[1].get_weights()[2]
        #W1 = self.model.layers[1].get_weights()[0]
        
        output_inter = self.model.get_layer('snn_layer_conv').output
        #print('DEBUG',tf.shape(output_inter))
        inter_model = tf.keras.models.Model(inputs=self.model.input, outputs=output_inter)
        #print('DEBUG',tf.shape(self.model.input))
        inter_pred=inter_model.predict(self.data, batch_size=10)
        print('DEBUG',inter_pred.shape[0], inter_pred.shape[1],inter_pred.shape[2] )
        print('DEBUG', np.max(inter_pred[0,:,:,0]), np.min(inter_pred[0,:,:,0]))
        plt.imshow(inter_pred[0,:,:,0])
        plt.colorbar()
        plt.show()
        # plt.imshow(inter_pred[1001,:,:,0])
        # plt.colorbar()
        # plt.show()
        # plt.imshow(inter_pred[12451,:,:,0])
        # plt.colorbar()
        # plt.show()






    
########################################################################################################
########################################################################################################
#%% Define sparseConnect Model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, MaxPooling2D, Flatten, Conv2D


N_in = np.shape(x_train)[-2]
N_channel = np.shape(x_train)[-1]

x = Input(shape=(N_in, N_in, N_channel), batch_size=10)
x_ = snn_layer_conv()(x)
x_ = MaxPooling2D((2, 2), strides=(2, 2))(x_)
x_ = Flatten()(x_)

x_ = Dense(512,activation='relu')(x_)
y = Dense(10,activation='softmax')(x_)

model = Model(inputs=x, outputs=y)
model.summary()

callbacks = [sample_vis(x_train)]

#define optimizer and learning rate
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)


#model compilation
model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

history = model.fit(x=x_train, y=y_train,
                            batch_size = 10,
                             epochs=10,
                             callbacks = callbacks,
                             verbose=1)
                            # callbacks = callbacks)

########################################################################################################
########################################################################################################
#%% Start training



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
   
        