# -*- coding: utf-8 -*-
"""
##########################################################################################################
Author      : l.gonzalez.carabarin@tue.nl
Institution : Eindhoven University of Technology 
Description : Supervised training of SNN layers based on TF 
Date        : 17-06-2022

##########################################################################################################
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import numpy as np
import time as mytime
from layers_tf import snn_layer_conv
from callbacks import sample_vis


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


N_in = np.shape(x_train)[-2]
N_channel = np.shape(x_train)[-1]

x = Input(shape=(N_in, N_in, N_channel), batch_size=10)
x_= snn_layer_conv(filters=2, channels = 1, name = 'snn_conv_1')(x)
x_ = snn_layer_conv(filters=4, channels = 2, name = 'snn_conv_2')(x_)
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
                             epochs=15,
                             #callbacks = callbacks,
                             verbose=1)


########################################################################################################
########################################################################################################

        