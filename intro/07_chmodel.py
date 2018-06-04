# # Simulation of channel model

import tensorflow as tf
import numpy as np

np.random.seed( 0 )

N = 2

x = tf.placeholder( dtype = tf.float32, shape = (None), name = "x" )
H = tf.placeholder( dtype = tf.float32, shape = (None), name = "H" )
n = tf.placeholder( dtype = tf.float32, shape = (None), name = "n" )

# y = tf.add( tf.multiply(H,x), n, name = "y" )
y = tf.reduce_sum( H*x[:,:,None], axis=1 )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    x_  = np.random.randn( N, 2 )

    # H_  = np.random.randn( N, 2, 2 )
    # H_[:,:,1] = H_[:,::-1,0]
    # H_[:,0,1] = -H_[:,0,1]
    H_  = np.eye( 2 )
    H_  = np.repeat(H_[np.newaxis,:,:], N, axis = 0 )
    n_  = np.random.randn( N, 2 )
    n_ *= 0

    # print( "shape(x) = {}".format(sess.run(tf.shape(x),feed_dict={x:x_,H:H_,n:n_})) )
    # print( "shape(H) = {}".format(sess.run(tf.shape(H),feed_dict={x:x_,H:H_,n:n_})) )
    # print( "shape(n) = {}".format(sess.run(tf.shape(n),feed_dict={x:x_,H:H_,n:n_})) )
    # print( "shape(y) = {}".format(sess.run(tf.shape(y),feed_dict={x:x_,H:H_,n:n_})) )

    y_  = sess.run( y, feed_dict={x:x_,H:H_,n:n_} )
    print( "x = {:s}".format(x_) )
    print( "H = {:s}".format(H_) )
    print( "n = {:s}".format(n_) )
    print( "y = {:s}".format(y_) )

    # print( "x[:,:,None] = {}".format(sess.run(x[:,:,None],feed_dict={x:x_,H:H_,n:n_})) )
    # print( "H = {}".format(sess.run(H,feed_dict={x:x_,H:H_,n:n_})) )
    # print( "H*x[:,:,None] = {}".format(sess.run(H*x[:,:,None],feed_dict={x:x_,H:H_,n:n_})) )
    # print( "x[:,:,None]*H = {}".format(sess.run(x[:,:,None]*H,feed_dict={x:x_,H:H_,n:n_})) )


# import tensorflow as tf

# N = 5
# C = 2

# Z = tf.ones((N, C, C))  # Shape=(N, C, C)
# x = tf.reshape(tf.range(0, N*C, dtype=tf.float32), shape=(N, C))  # Shape=(N, C)
# mul = tf.reduce_sum(Z * x[:, :, None], axis=-1)  # Shape=(N, C)

# with tf.Session() as sess:
#     print( sess.run( Z ) )
#     print( sess.run( x ) )
#     print(sess.run(mul))