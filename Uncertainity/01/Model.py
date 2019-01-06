
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def model( input, outdim, dropRate, trainMode, reuse = False ):
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
            net = tf.layers.conv2d( net, 64, [3,3],
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
            net = tf.layers.dropout( net, training = trainMode, rate = dropRate )

        with tf.variable_scope("fc2") as scope:
            net = tf.layers.dense( net, 32,
                                   activation = tf.nn.relu,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                   reuse = reuse )
            net = tf.layers.dropout( net, training = trainMode, rate = dropRate )

        with tf.variable_scope("out") as scope:
            net = tf.layers.dense( net, outdim,
                                   activation = tf.nn.softmax,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                   reuse = reuse )
    return net

if __name__ == '__main__':
    from tensorflow.keras.datasets import mnist

    nClasses = 10
    (trainX, trainY), (testX,testY) = mnist.load_data()
    trainX  = np.expand_dims(trainX,axis=3) / 255.0
    testX   = np.expand_dims(testX,axis=3) / 255.0
    inShape = [None] + list( trainX.shape[1:] )
    outDim  = [ None, nClasses ]

    inputTensor = tf.placeholder( tf.float32, inShape, name = "input" )
    labelTensor = tf.placeholder( tf.float32, outDim, name = "output" )
    dropRate    = tf.placeholder( tf.float32, name = "drop_rate" )
    trainFlag   = tf.placeholder_with_default( True, [], name = "train_mode" )

    prediction = model( inputTensor,  nClasses, dropRate, trainFlag, reuse = False )

    tvars = tf.trainable_variables()
    print( tvars )
    
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )

        img     = trainX[:1,]
        label   = np.eye(nClasses)[trainY[:1]]
        print( sess.run( prediction, 
                        feed_dict = { inputTensor: img, 
                                      labelTensor: label, 
                                      dropRate   : 1.0,
                                      trainFlag  : True } ) )
        print( sess.run( prediction, 
                        feed_dict = { inputTensor: img, 
                                      labelTensor: label, 
                                      dropRate   : 1.0 } ) )
        print( sess.run( prediction, 
                        feed_dict = { inputTensor: img, 
                                      labelTensor: label, 
                                      dropRate   : 1.0,
                                      trainFlag  : False } ) )  