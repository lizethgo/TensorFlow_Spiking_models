# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
##########################################################################################################
Author      : l.gonzalez.carabarin@tue.nl
Institution : Eindhoven University of Technology 
Description : Supervised training of SNN layers based on TF. This file contains the callback to 
              visualize resulting filters.
Date        : 17-06-2022

##########################################################################################################
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import numpy as np



class sample_vis(tf.keras.callbacks.Callback):
    def __init__(self, data, **kwargs):
        self.data = data
    def on_epoch_begin(self, epoch, logs=None):

        
        output_inter = self.model.get_layer('snn_conv_1').output
        inter_model = tf.keras.models.Model(inputs=self.model.input, outputs=output_inter)
        inter_pred =inter_model.predict(self.data, batch_size=10)


        plt.imshow(inter_pred[0,:,:,0])
        plt.colorbar()
        plt.show()
        # plt.imshow(inter_pred[1001,:,:,0])
        # plt.colorbar()
        # plt.show()
        plt.imshow(inter_pred[12451,:,:,0])
        plt.colorbar()
        plt.show()
