# Uncertainity using drop out

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime

import Dataset
import Model


nClasses = 6

ds = Dataset.MNISTDataset( nClasses )


inShape = [None] + list( ds.trainX.shape[1:] )
outDim  = [ None, nClasses ]

inputTensor = tf.placeholder( tf.float32, inShape, name = "input" )
labelTensor = tf.placeholder( tf.float32, outDim, name = "output" )
keepTensor  = tf.placeholder( tf.float32, name = "keep_rate" )

prediction = Model.model( inputTensor, keepTensor, nClasses, reuse = False )

with tf.name_scope("trainer"):
    xEnt    = tf.reduce_mean( -tf.reduce_sum(labelTensor*tf.log(prediction),reduction_indices=[1]) )
    objective   = tf.train.AdamOptimizer(0.01).minimize(xEnt)

logsPath   = './models/' + \
                        "M_{:04d}x{:02d}_".format(np.prod(inShape[1:]),nClasses) + \
                        datetime.datetime.now().strftime("%Y%m%d%H%M%S")

builder = tf.saved_model.builder.SavedModelBuilder( logsPath+"/" )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    writer  = tf.summary.FileWriter( logsPath )
    writer.add_graph( sess.graph )


    for i in range(25):
        img, label  = ds.get_train_batch( 64 )
        l, _ = sess.run( [xEnt,objective], feed_dict = { inputTensor: img, labelTensor: label, keepTensor: 0.80 } )
        print( "i = ", i , "   l = ", l )
    
    writer.close()
    # Save model
    builder.add_meta_graph_and_variables( sess,
            [ tf.saved_model.tag_constants.TRAINING ],
            signature_def_map = None,
            assets_collection = None )
    builder.save()