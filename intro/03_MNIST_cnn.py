# A 2-layed MNIST classifier

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

logs_path   = "./03_MNIST_cnn/"

nEpochs     = 1000
batchSize   = 2500
nChkPt      = 100


mnist   = input_data.read_data_sets( "MNIST_data/", one_hot = True, 
                                     validation_size = 1000 )
print( "Train      : %d" %(mnist.train.num_examples) )
print( "Validation : %d" %(mnist.validation.num_examples) )
print( "Test       : %d" %(mnist.test.num_examples) )

# For fully connected layer
W  = tf.Variable( tf.random_normal([784,10],stddev=0.1), name = "FC_Weight" )
b  = tf.Variable( tf.random_normal([10],stddev=0.1), name = "FC_bias" )
# Filters
l1_filts    = tf.Variable( tf.truncated_normal([4,4,1,4],stddev=0.1), name = "conv1_filters" )


X = tf.placeholder( dtype = tf.float32, shape = (None,784), name = "img" )
X_img = tf.reshape( X, [-1,28,28,1], name = "img_28x28x1" )
y = tf.placeholder( dtype = tf.float32, name = "label" )

# Make prediction graph
l1_out  = tf.nn.relu( tf.nn.conv2d(X_img,l1_filts,[1,2,2,1],"SAME") )

l1_out_flat = tf.reshape( l1_out, [-1,784] )
l2_out  = tf.nn.softmax( tf.matmul(l1_out_flat,W)+b )

xEnt    = tf.reduce_mean( -tf.reduce_sum(y*tf.log(l2_out),reduction_indices=[1]) )

# Make train graph
train   = tf.train.GradientDescentOptimizer( 0.5 ).minimize( xEnt )
numCorr = tf.equal( tf.argmax(l2_out,1), tf.argmax(y,1) )
acc     = tf.reduce_mean( tf.cast(numCorr,tf.float32) )

# Create a summary
# tf.scalar_summary( "xEnt", xEnt )
# tf.scalar_summary( "accuracy", acc )
# summary_op  = tf.merge_all_summaries()

# Session!!
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    writer  = tf.summary.FileWriter( logs_path, sess.graph )

    print( "Epoch    Train(T-1)    Train(T)    Validation" )
    print( "-----    ----------    --------    ----------" )
    for i in range(nEpochs):
        _img, _ans  = mnist.train.next_batch( batchSize )
        # if( i == 0 ):
        #     print( sess.run(tf.shape(X),feed_dict={X:_img}) )
        if( i%nChkPt == 0 ):
            # Run accuracy graph with this input to check
            thisEpochPreAcc     = sess.run( acc, feed_dict={X:_img,y:_ans} )
            # Update graph parameters
        sess.run( train, feed_dict={X:_img,y:_ans} )
        if( i%nChkPt == 0 or i==nEpochs-1):
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