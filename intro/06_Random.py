# Injecting noise at the middle of the network

import tensorflow as tf
import numpy as np

ip_dim  = 5
op_dim  = 2
samples = 3

x = np.random.randn(samples, ip_dim )

print( "x = " ),
print( x )

X = tf.placeholder( shape = (None,ip_dim), dtype = tf.float32, name = "ip" )

with tf.name_scope( "Encoder" ):
    W1  = tf.Variable( tf.random_normal([ip_dim,op_dim],stddev=0.1) )
    b1  = tf.Variable( tf.random_normal([op_dim],stddev=0.1) )
    enc_out = tf.add( tf.matmul(X,W1), b1, name = "enc_out" )
with tf.name_scope( "Corruption" ):
    noise   = tf.random_normal( shape = tf.shape(enc_out),
                    mean = 0.00, stddev = 0.1, dtype = tf.float32 )
    cor_out = tf.add( enc_out, noise, name = "cor_out" )
with tf.name_scope( "Decoder" ):
    W2  = tf.Variable( tf.random_normal([op_dim,ip_dim],stddev=0.1) )
    b2  = tf.Variable( tf.random_normal([ip_dim],stddev=0.1) )
    dec_out = tf.add( tf.matmul(cor_out,W2), b2, name = "dec_out" )

loss    = tf.nn.l2_loss( dec_out-X )
opt     = train   = tf.train.GradientDescentOptimizer( 0.5 ).minimize( loss )

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    for i in range(3):
        print( "Pass {}".format(i) )
        o_W1, o_b1, o_W2, o_b2 = sess.run( [W1,b1,W2,b2] )
        print( "W1 = {:s}\nb1 = {:s}\nW2 = {:s}\nb2 = {:s}".format(o_W1,o_b1,o_W2,o_b2) )
        x1, x2, x3 = sess.run( [enc_out,cor_out,dec_out], feed_dict = {X:x} )
        print( "enc_out = {:s}\ncor_out = {:s}\n".format(x1,x2) )
