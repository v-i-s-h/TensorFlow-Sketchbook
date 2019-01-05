
# https://github.com/ardiya/siamesenetwork-tensorflow/blob/master/model.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

flags   = tf.app.flags
FLAGS   = flags.FLAGS

def model( input, keeprate, outdim, reuse = False ):
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
                                    activation = tf.nn.relu,
                                    padding = 'SAME',
                                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                    # scope = scope,
                                    reuse = reuse )
            net = tf.layers.max_pooling2d( net, [2,2], 2, padding = 'SAME' )

        with tf.variable_scope("flatten") as scope:
            net = tf.layers.flatten( net )

        with tf.variable_scope("fc1") as scope:
            net = tf.layers.dense( net, 64, 
                                   activation = tf.nn.relu,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                   reuse = reuse )
            net = tf.layers.dropout( net, rate = keeprate )

        with tf.variable_scope("fc2") as scope:
            net = tf.layers.dense( net, 32,
                                   activation = tf.nn.relu,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                   reuse = reuse )
            net = tf.layers.dropout( net, rate = keeprate )

        with tf.variable_scope("out") as scope:
            net = tf.layers.dense( net, outdim,
                                   activation = tf.nn.softmax,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                   reuse = reuse )
    return net