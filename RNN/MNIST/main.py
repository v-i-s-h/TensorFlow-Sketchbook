# Source: https://www.knowledgemapper.com/knowmap/knowbook/jasdeepchhabra94@gmail.comUnderstandingLSTMinTensorflow(MNISTdataset)

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data

mnist   = input_data.read_data_sets( "/tmp/data/", one_hot = True  )

time_steps  = 28
num_units   = 128
n_input     = 28
learning_rate   = 0.001
n_classes   = 10
batch_size  = 128

out_weights = tf.Variable( tf.random_normal([num_units,n_classes]) )
out_bias    = tf.Variable( tf.random_normal([n_classes]) )

x   = tf.placeholder( "float", [None,time_steps,n_input] )
y   = tf.placeholder( "float", [None,n_classes] )

input   = tf.unstack( x, time_steps, 1 )
lstm_layer  = rnn.BasicLSTMCell( num_units, forget_bias = 1 )
outputs, state  = rnn.static_rnn( lstm_layer, input, dtype = "float32" )

prediction  = tf.matmul( outputs[-1], out_weights ) + out_bias

loss    = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
opt     = tf.train.AdamOptimizer( learning_rate = learning_rate ).minimize(loss)

correct_prediction  = tf.equal( tf.argmax(prediction,1), tf.argmax(y,1) )
accuracy    = tf.reduce_mean( tf.cast(correct_prediction,tf.float32) )

init    = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run( init )
    iter = 1
    while iter <= 100:
        batch_x, batch_y    = mnist.train.next_batch( batch_size = batch_size )
        
        batch_x = batch_x.reshape( (batch_size,time_steps,n_input) )
        
        sess.run( opt, feed_dict = {x:batch_x,y:batch_y} )

        if iter%10 == 0:
            acc     = sess.run( accuracy, feed_dict = {x:batch_x,y:batch_y} )
            los     = sess.run( loss, feed_dict = {x:batch_x,y:batch_y} )
            print( "Iter: {:4d}    Accuracy: {:.3f}    Loss: {:.3f}".format(iter,acc,los) )
        
        iter = iter + 1
    
    print( "Entering test phase..." )
    test_x, test_y  = mnist.test.next_batch( batch_size = 1 )
    test_x  = test_x.reshape( (1,time_steps,n_input) )
    print( "Test y ={}".format(test_y) )
    this_out, this_state, this_prediction    = sess.run( [outputs,state,prediction], feed_dict={x:test_x,y:test_y} )
    print( "this_out  = {}".format(this_out) )
    print( "this_state = {}".format(this_state) )
    print( "this_prediction = {}".format(this_prediction) )


