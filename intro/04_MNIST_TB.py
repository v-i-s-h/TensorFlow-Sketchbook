# A 2-layed MNIST classifier

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

logs_path   = "./04_MNIST_TB/"

nEpochs     = 10000
batchSize   =  500
nDispChkPt  =  1000
nSummChkPt  =   250


mnist   = input_data.read_data_sets( "MNIST_data/", one_hot = True, 
                                     validation_size = 1000 )
print( "Train      : %d" %(mnist.train.num_examples) )
print( "Validation : %d" %(mnist.validation.num_examples) )
print( "Test       : %d" %(mnist.test.num_examples) )


with tf.name_scope( "Input_layer" ):
    X = tf.placeholder( dtype = tf.float32, shape = (None,784), name = "img" )
    X_img = tf.reshape( X, [-1,28,28,1], name = "img_28x28x1" )
    y = tf.placeholder( dtype = tf.float32, name = "label" )
    tf.summary.image( "Input", X_img, 3 )

with tf.name_scope( "L1_conv" ):
    l1_filts    = tf.Variable( tf.truncated_normal([4,4,1,4],stddev=0.1), name = "conv1_filters" )
    l1_bias = tf.Variable( tf.truncated_normal([4],stddev=0.1), name = "conv1_bias" )
    l1_out  = tf.nn.relu( tf.nn.conv2d(X_img,l1_filts,[1,2,2,1],"SAME")+l1_bias )
    tf.summary.histogram( "L1_filter_weights", l1_filts )
    tf.summary.histogram( "L1_bias", l1_bias )
    tf.summary.histogram( "L1_act", l1_out )

with tf.name_scope( "L2_conv" ):
    l2_filts    = tf.Variable( tf.truncated_normal([3,3,4,4],stddev=0.1), name = "conv2_filters" )
    l2_bias = tf.Variable( tf.truncated_normal([4],stddev=0.1), name = "conv2_bias" )
    l2_out  = tf.nn.relu( tf.nn.conv2d(l1_out,l2_filts,[1,1,1,1],"SAME")+l2_bias )
    tf.summary.histogram( "L2_filter_weights", l2_filts )
    tf.summary.histogram( "L2_bias", l2_bias )
    tf.summary.histogram( "L2_act", l2_out )

l2_out_flat = tf.reshape( l2_out, [-1,784] )

# For fully connected layer
with tf.name_scope( "FC_out" ):
    W  = tf.Variable( tf.random_normal([784,10],stddev=0.1), name = "FC_Weight" )
    b  = tf.Variable( tf.random_normal([10],stddev=0.1), name = "FC_bias" )
    l3_out  = tf.nn.softmax( tf.matmul(l2_out_flat,W)+b )
    tf.summary.histogram( "FC_weights", W )
    tf.summary.histogram( "FC_bias", b )
    tf.summary.histogram( "FC_act", l3_out )

with tf.name_scope( "Eval" ):
    xEnt    = tf.reduce_mean( -tf.reduce_sum(y*tf.log(l3_out),reduction_indices=[1]),
                                name = "xEnt" )
    numCorr = tf.equal( tf.argmax(l3_out,1), tf.argmax(y,1) )
    acc     = tf.reduce_mean( tf.cast(numCorr,tf.float32) )

# Make train graph
with tf.name_scope( "Train" ):
    train   = tf.train.GradientDescentOptimizer(0.2).minimize(xEnt)

# Create a summary
tf.summary.scalar( "xEnt", xEnt )
tf.summary.scalar( "accuracy", acc )
summary_all = tf.summary.merge_all()

# Session!!
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    writer  = tf.summary.FileWriter( logs_path + 
                    datetime.datetime.now().strftime("%Y%m%d%H%M%S") )
    writer.add_graph( sess.graph )

    print( "Epoch    Train(T-1)    Train(T)    Validation" )
    print( "-----    ----------    --------    ----------" )
    for i in range(nEpochs):
        _img, _ans  = mnist.train.next_batch( batchSize )
        # if( i == 0 ):
        #     print( sess.run(tf.shape(X),feed_dict={X:_img}) )
        if( i%nDispChkPt == 0 ):
            # Run accuracy graph with this input to check
            thisEpochPreAcc     = sess.run( acc, feed_dict={X:_img,y:_ans} )
        if( i%nSummChkPt == 0 ):
            s = sess.run( summary_all, feed_dict = {X:_img,y:_ans} )
            writer.add_summary( s, i )
        # Update graph parameters
        sess.run( train, feed_dict={X:_img,y:_ans} )
        if( i%nDispChkPt == 0 or i==nEpochs-1):
            # Recalculate accuracy
            thisEpochBatchAcc   = sess.run( acc, feed_dict={X:_img,y:_ans} )
            # Calculate accuracy for validation batch
            _img, _ans  = mnist.validation.images, mnist.validation.labels
            thisEpochValAcc = sess.run( acc, feed_dict={X:_img,y:_ans})
            print( "%5d      %7.6f    %7.6f      %7.6f" 
                %(i,thisEpochPreAcc,thisEpochBatchAcc,thisEpochValAcc) )
    print( "-----    ----------    --------    ----------" )

    # Test 
    _img, _ans  = mnist.test.images, mnist.test.labels
    print( "\n    Test Accuracy: %r\n" 
                %(sess.run(acc,feed_dict={X:_img,y:_ans})) )

    writer.close()