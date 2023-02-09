# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:25:05 2022

@author: 20195088
"""

# -*- coding: utf-8 -*-
"""
#################################################################################
custom layers
from custom_layer import my
my_dense_layer(units, ativation)

#################################################################################   
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda, Activation
from tensorflow.keras import regularizers
import numpy as np

########################################################################################################
########################################################################################################
# Activate the following lines for GPU's usage, comment these lines if no GPU's are avilable

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
# # Currently, memory growth needs to be the same across GPUs
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# tf.config.experimental.set_visible_devices(gpus[2],'GPU')


##########################################################################################################################
############################################################################################################################



class my_dense_layer(Layer):   
    def __init__(self,units, activation=None, name=None, **kwargs):
        self.units = units
        self.activation = activation
        super(my_dense_layer, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units':self.units, 
            'activation':self.activation,

        })
        return config

    def build(self, input_shape): 
        # Define weight matrix and bias vector
        self.W = self.add_weight(shape=[int(input_shape[-1]),self.units],
                                 initializer='he_uniform',
                                 #regularizer=regularizers.l2(0.0005),
                                 trainable=True, name='w', dtype='float32')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True, name='b',  dtype='float32')
        


        super(my_dense_layer, self).build(input_shape) 
 
    def call(self, x):
        units = self.units
   
        # Produce layer output

        y = tf.add(tf.matmul(x, self.W),self.b)
        if not self.activation == None:
            y = Activation(self.activation)(y)
        
        return y
        
        
class my_conv2D(Layer):   
    def __init__(self, kernel_size, filters,channel_size, activation=None, name=None, **kwargs):
        self.activation = activation
        self.kernel_size = kernel_size
        self.filters = filters
        self.channel_size = channel_size
       
        super(my_conv2D, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation':self.activation,
            'kernel_size':self.kernel_size,
            'filters':self.filters,
            'channel_size':self.channel_size

        })
        return config

    def build(self, input_shape): 
        # Define weight matrix and bias vector
        self.k = self.add_weight(shape=[self.kernel_size,self.kernel_size, self.channel_size, self.filters],
                                 initializer='he_uniform',
                                 #regularizer=regularizers.l2(0.0005),
                                 trainable=True, name='w', dtype='float32')
        self.b = self.add_weight(shape=(self.filters,),
                                 initializer='zeros',
                                 trainable=True, name='b',  dtype='float32')
        


        super(my_conv2D, self).build(input_shape) 
 
    def call(self, x):
        # Produce layer output
        y = tf.nn.conv2d(x, self.k, strides=1, padding = 'SAME')
        print(np.shape(y))
        y = tf.nn.bias_add(y,self.b)  
        
        return y


class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    #print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
    print(self.model.layers[1])


  # def on_train_batch_end(self, batch, logs=None):
  #   #print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  # def on_test_batch_begin(self, batch, logs=None):
  #   #print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  # def on_test_batch_end(self, batch, logs=None):
  #   #print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
 

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
#%% Define sparseConnect Model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, MaxPooling2D, Flatten, Conv2D

# Parameters
n_epochs = 20

N_in = np.shape(x_train)[-2]
N_channel = np.shape(x_train)[-1]
N_out = np.size(y_train,-1)

x_ = Input(shape=(N_in, N_in, N_channel), batch_size=64)
x = my_conv2D(kernel_size=3, filters=20, channel_size = 1, activation='relu')(x_)
#x = Cconv2D(20, kernel_size = (3,3), strides=1, padding = 'SAME')(x_)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = my_conv2D(kernel_size=3,filters=50, channel_size = 20, activation='relu')(x)
#x = Conv2D(50, kernel_size = (3,3), strides=1, padding = 'SAME')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Flatten()(x)

x = my_dense_layer(512,activation='relu')(x)
y = my_dense_layer(10,activation='softmax')(x)
model = Model(inputs=x_, outputs=y)

model.summary()
  
########################################################################################################
########################################################################################################

#define optimizer and learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

callbacks = [
             MyCustomCallback()]


#model compilation
model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
              
callbacks = callbacks
history = model.fit(x=x_train, y=y_train,
                            batch_size = 64,
                            epochs=2,
                            validation_data=(x_val, y_val),verbose=1)

# validation_data=(x_val, y_val)
results = model.evaluate(x_test, y_test, batch_size=64)
              
# callback_model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
  # filepath='lenet5_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
  # monitor = "val_loss",
  # verbose = 0,
  # save_best_only = False,
  # save_weights_only = False,
  # save_freq = "epoch"
# )
# callbacks = [callback_model_checkpoint]
              
# # #%% Start training
# #history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=1, verbose=1,
# #         validation_data=(x_val,y_val), callbacks=callbacks)
    




# def get_flops(model):
    # session = tf.compat.v1.Session()
    # graph = tf.compat.v1.get_default_graph()
        

    # with graph.as_default():
        # with session.as_default():
            # model = tf.keras.models.load_model(model, custom_objects={'my_dense_layer':my_dense_layer}) # 'my_conv2D':my_conv2D

            # run_meta = tf.compat.v1.RunMetadata()
            # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # # We use the Keras session graph in the call to the profiler.
            # flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  # run_meta=run_meta, cmd='op', options=opts)
        
            # return flops.total_float_ops

# flops = get_flops('lemet5_weights.01-0.05.hdf5')
# print(flops)
