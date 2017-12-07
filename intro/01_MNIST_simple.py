# Simple MNIST network: Simple Logistic regression

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist   = input_data.read_data_sets( "MNIST_data/", one_hot = True, 
                                     validation_size = 1000 )

print( "Train      : %d" %(mnist.train.num_examples) )
print( "Validation : %d" %(mnist.validation.num_examples) )
print( "Test       : %d" %(mnist.test.num_examples) )

batchSize   = 2500
W   = tf.Variable( tf.random_normal([784,10],stddev=0.1) )
b   = tf.Variable( tf.random_normal([10],stddev=0.1) )

img = tf.placeholder( dtype = tf.float32, name = "img" )
ans = tf.placeholder( dtype = tf.float32, name = "label" )

# Make prediction graph
prob    = tf.nn.softmax( tf.matmul(img,W) + b )
xEnt    = tf.reduce_mean( -tf.reduce_sum(ans*tf.log(prob),reduction_indices=[1]) )

# Make train graph
train   = tf.train.GradientDescentOptimizer( 0.5 ).minimize( xEnt )
numCorr = tf.equal( tf.argmax(prob,1), tf.argmax(ans,1) )
acc     = tf.reduce_mean( tf.cast(numCorr,tf.float32) )

# Session!!
sess    = tf.Session()
sess.run( tf.global_variables_initializer() )

nEpochs = 10000
print( "Epoch    Train(T-1)    Train(T)    Validation" )
print( "-----    ----------    --------    ----------" )
for i in range(nEpochs):
    _img, _ans  = mnist.train.next_batch( batchSize )
    if( i%100 == 0 ):
        # Run accuracy graph with this input to check
        thisEpochPreAcc     = sess.run( acc, feed_dict={img:_img,ans:_ans} )
        # Update graph parameters
    sess.run( train, feed_dict={img:_img,ans:_ans} )
    if( i%100 == 0 or i==nEpochs-1):
        # Recalculate accuracy
        thisEpochBatchAcc   = sess.run( acc, feed_dict={img:_img,ans:_ans} )
        # Calculate accuracy for validation batch
        _img, _ans  = mnist.validation.images, mnist.validation.labels
        thisEpochValAcc = sess.run( acc, feed_dict={img:_img,ans:_ans})
        print( "%5d      %7.6f    %7.6f      %7.6f" 
            %(i,thisEpochPreAcc,thisEpochBatchAcc,thisEpochValAcc) )
print( "-----    ----------    --------    ----------" )

# Test 
_img, _ans  = mnist.test.images, mnist.test.labels
print( "\n    Test Accuracy: %r\n" 
            %(sess.run(acc,feed_dict={img:_img,ans:_ans})) )