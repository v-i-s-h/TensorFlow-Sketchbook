
# https://github.com/ardiya/siamesenetwork-tensorflow/blob/master/model.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

flags   = tf.app.flags
FLAGS   = flags.FLAGS

def mnist_model( input, reuse = False ):
    with tf.name_scope( "model" ):
        with tf.variable_scope("conv1") as scope:
            net = tf.layers.conv2d( input, 32, [7,7],
                                    activation = tf.nn.relu,
                                    padding = 'SAME',
                                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                    # scope = scope,
                                    reuse = reuse )
            net = tf.layers.max_pooling2d( net, [2,2], 2, padding = 'SAME' )
        
        with tf.variable_scope("conv2") as scope:
            net = tf.layers.conv2d( net, 64, [5,5],
                                    activation = tf.nn.relu,
                                    padding = 'SAME',
                                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                    # scope = scope,
                                    reuse = reuse )
            net = tf.layers.max_pooling2d( net, [2,2], 2, padding = 'SAME' )

        with tf.variable_scope("conv3") as scope:
            net = tf.layers.conv2d( net, 128, [3,3],
                                    activation = tf.nn.relu,
                                    padding = 'SAME',
                                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                    # scope = scope,
                                    reuse = reuse )
            net = tf.layers.max_pooling2d( net, [2,2], 2, padding = 'SAME')
        
        with tf.variable_scope("conv4") as scope:
            net = tf.layers.conv2d( net, 2, [1,1],
                                    activation = None,
                                    padding = 'SAME',
                                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                    # scope = scope,
                                    reuse = reuse )
            net = tf.layers.max_pooling2d( net, [2,2], 2, padding = 'SAME' )

        with tf.variable_scope"flatten") as scope:
            net = tf.layers.flatten( net )

        with tf.variable_scope

    return net

def contrastive_loss( model1, model2, y, margin ):
    with tf.name_scope("contrastive_loss"):
        distance    = tf.sqrt( tf.reduce_sum(tf.pow(model1-model2,2),1,keepdims=True) )
        simillarity = y * distance
        dissimilarity   = (1-y) * tf.square( tf.maximum((margin-distance),0) )
        return tf.reduce_mean( dissimilarity + simillarity ) / 2.0