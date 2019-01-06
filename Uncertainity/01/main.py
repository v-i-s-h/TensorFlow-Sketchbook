# Uncertainity using drop out

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime

import Dataset
import Model


nClasses = 7
nEpochs  = 20

ds = Dataset.MNISTDataset( nClasses )

inShape = [None] + list( ds.trainX.shape[1:] )
outDim  = [ None, nClasses ]

with tf.name_scope("input"):
    inputTensor = tf.placeholder( tf.float32, inShape, name = "input" )
    labelTensor = tf.placeholder( tf.float32, outDim, name = "output" )
    dropRate    = tf.placeholder_with_default( 0.0, [], name = "drop_rate" )
    trainMode   = tf.placeholder_with_default( True, [], name = "train_mode" )

prediction = Model.model( inputTensor, nClasses, dropRate, trainMode, reuse = False )

with tf.name_scope("trainer"):
    xEnt    = tf.reduce_mean( -tf.reduce_sum(labelTensor*tf.log(prediction+1e-20),reduction_indices=[1]) )
    objective   = tf.train.AdamOptimizer(0.01).minimize(xEnt)

with tf.name_scope("eval"):
    numCorr = tf.equal( tf.argmax(prediction,1), tf.argmax(labelTensor,1) )
    acc     = tf.reduce_mean( tf.cast(numCorr,tf.float32) )

logsPath   = './models/' + \
                        "M_{:04d}x{:02d}_".format(np.prod(inShape[1:]),nClasses) + \
                        datetime.datetime.now().strftime("%Y%m%d%H%M%S")

builder = tf.saved_model.builder.SavedModelBuilder( logsPath+"/" )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    writer  = tf.summary.FileWriter( logsPath )
    writer.add_graph( sess.graph )

    for i in range(nEpochs):
        trainLoss = []
        testLoss  = []
        testAcc   = []
        # Run one-pass over train set
        for (img, label) in ds.get_train_batch( 64 ):
            l, _ = sess.run( [xEnt,objective], feed_dict = { inputTensor: img, 
                                                             labelTensor: label, 
                                                             dropRate   : 0.1 } )
            trainLoss.append(l)
        # Run one evaluation over testset
        for (img,label) in ds.get_test_batch(512):
            l, a = sess.run( [xEnt,acc], feed_dict = { inputTensor: img,
                                                       labelTensor: label,
                                                       dropRate   : 0.5 } )
            testLoss.append(l)
            testAcc.append(a)
        print( "{:4d}: {:10.4f}    {:10.4f}    {:10.4f}".format(i,np.mean(trainLoss),np.mean(testLoss),np.mean(testAcc)) )
    
    writer.close()
    # Save model
    builder.add_meta_graph_and_variables( sess,
            [ tf.saved_model.tag_constants.TRAINING ],
            signature_def_map = None,
            assets_collection = None )
    builder.save()