# A 2-layed MNIST classifier

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import numpy as np

# Setup Custom Module Path
import os, sys
sys.path.append( './../' )
import Optimizers
import Optimizers3

(xTrain,yTrain), (xTest,yTest) = mnist.load_data()
xTrain = np.reshape( xTrain, [-1,np.prod(xTrain.shape[1:])] )
xTest = np.reshape( xTest, [-1,np.prod(xTrain.shape[1:])] )
xTrain = xTrain.astype('float32') / 255.0
xTest = xTest.astype('float32') / 255.0
yTrain = keras.utils.to_categorical(yTrain, 10)
yTest = keras.utils.to_categorical(yTest, 10)

batchSize   = 2500
W1  = tf.Variable( tf.random_normal([784,100],stddev=0.1) )
b1  = tf.Variable( tf.random_normal([100],stddev=0.1) )
W2  = tf.Variable( tf.random_normal([100,10],stddev=0.1) )
b2  = tf.Variable( tf.random_normal([10],stddev=0.1) )

X = tf.placeholder( dtype = tf.float32, name = "img" )
y = tf.placeholder( dtype = tf.float32, name = "label" )

# Make prediction graph
l1_out  = tf.nn.relu( tf.matmul(X,W1)+b1 )
l2_out  = tf.nn.softmax( tf.matmul(l1_out,W2)+b2 )

xEnt    = tf.reduce_mean( -tf.reduce_sum(y*tf.log(l2_out+1e-10),reduction_indices=[1]) )

# Make train graph
# train   = tf.train.GradientDescentOptimizer( 0.5 ).minimize( xEnt )
# train = Optimizers.PowerSignOptimizer(learning_rate=0.0001).minimize(xEnt)
train = Optimizers3.SimultaneousPerturbationOptimizer(a=0.05,c=0.05,alpha=0.99,gamma=0.40).minimize(xEnt)
# train = Optimizers3.MyOpt(learning_rate=0.5).minimize(xEnt)
numCorr = tf.equal( tf.argmax(l2_out,1), tf.argmax(y,1) )
acc     = tf.reduce_mean( tf.cast(numCorr,tf.float32) )

# Session!!
sess    = tf.Session()
sess.run( tf.global_variables_initializer() )

nEpochs = 10000
nChkPt  = 100
print( "Epoch    Train(T-1)    Train(T)    Validation" )
print( "-----    ----------    --------    ----------" )

start_pos = 0
batch_size = 64
for i in range(nEpochs):
    _img = xTrain[start_pos:start_pos+batch_size,:]
    _ans = yTrain[start_pos:start_pos+batch_size,:]
    start_pos += batch_size
    if start_pos+batch_size > 60000:
      start_pos = 0
    if( i%nChkPt == 0 ):
        # Run accuracy graph with this input to check
        thisEpochPreAcc     = sess.run( acc, feed_dict={X:_img,y:_ans} )
        # Update graph parameters
    sess.run( train, feed_dict={X:_img,y:_ans} )
    if( i%nChkPt == 0 or i==nEpochs-1):
        # Recalculate accuracy
        thisEpochBatchAcc   = sess.run( acc, feed_dict={X:_img,y:_ans} )
        # Calculate accuracy for validation batch
        thisEpochValAcc = sess.run( acc, feed_dict={X:xTest,y:yTest})
        print( "%5d      %7.6f    %7.6f      %7.6f" 
            %(i,thisEpochPreAcc,thisEpochBatchAcc,thisEpochValAcc) )
print( "-----    ----------    --------    ----------" )

# Test 
_img, _ans  = xTest, yTest
print( "\n    Test Accuracy: %r\n" 
            %(sess.run(acc,feed_dict={X:_img,y:_ans})) )