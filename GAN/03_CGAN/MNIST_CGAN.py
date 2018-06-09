# Conditional GAN for MNIST data

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist   = input_data.read_data_sets( "./../MNIST_data", one_hot = True )
mb_size = 64
Z_dim   = 100
X_dim   = mnist.train.images.shape[1]
y_dim   = mnist.train.labels.shape[1]
h_dim   = 128

print( "X_dim = {}    y_dim = {}".format(X_dim,y_dim) )

def xavier_init( size ):
    in_dim  = size[0]
    xavier_stddev   = 1.0 / tf.sqrt(in_dim/2.0)
    return tf.random_normal( shape = size, stddev = xavier_stddev )

# Discriminator
X   = tf.placeholder( tf.float32, shape = [None,784] )
y   = tf.placeholder( tf.float32, shape = [None,y_dim] )

D_W1    = tf.Variable( xavier_init([X_dim+y_dim,h_dim]) )
D_b1    = tf.Variable( tf.zeros(shape=[h_dim]) )
D_W2    = tf.Variable( xavier_init([h_dim,1]) )
D_b2    = tf.Variable( tf.zeros(shape=[1]) )
theta_D = [ D_W1, D_b1, D_W2, D_b2 ]

def discriminator( x, y ):
    inputs  = tf.concat( axis = 1, values = [x,y] )
    D_h1    = tf.nn.relu( tf.matmul(inputs,D_W1) + D_b1 )
    D_logit = tf.matmul( D_h1, D_W2 ) + D_b2
    D_prob  = tf.nn.sigmoid( D_logit )

    return D_prob, D_logit

# Generator
Z   = tf.placeholder( tf.float32, shape = [None,Z_dim] )

G_W1    = tf.Variable( xavier_init([Z_dim+y_dim,h_dim]) )
G_b1    = tf.Variable( tf.zeros(shape=[h_dim]) )
G_W2    = tf.Variable( xavier_init([h_dim,X_dim]) )
G_b2    = tf.Variable( tf.zeros(shape=[X_dim]) )
theta_G = [ G_W1, G_b1, G_W2, G_b2 ]

def generator( z, y ):
    inputs  = tf.concat( axis = 1, values = [z,y] )
    G_h1    = tf.nn.relu( tf.matmul(inputs,G_W1) + G_b1 )
    G_log_prob  = tf.matmul( G_h1, G_W2 ) + G_b2
    G_prob  = tf.nn.sigmoid( G_log_prob )

    return G_prob

def sample_Z( m, n ):
    return np.random.uniform( -1.0, +1.0, size = [m,n] )

def plot( samples ):
    fig = plt.figure( figsize=(4,4) )
    gs  = gridspec.GridSpec(4,4)
    gs.update( wspace = 0.05, hspace = 0.05 )

    for i,samples in enumerate(samples):
        ax  = plt.subplot( gs[i] )
        plt.axis( 'off' )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect( 'equal' )
        plt.imshow( samples.reshape(28,28), cmap = 'Greys_r' )

    return fig

G_sample   = generator( Z, y )
D_real, D_logit_real    = discriminator( X, y )
D_fake, D_logit_fake    = discriminator( G_sample, y )

D_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
                                logits = D_logit_real,
                                labels = tf.ones_like(D_logit_real)
                            ) )
D_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
                                logits = D_logit_fake,
                                labels = tf.zeros_like(D_logit_fake)
                            ) )
D_loss  = D_loss_real + D_loss_fake
G_loss  = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
                                logits = D_logit_fake,
                                labels = tf.ones_like(D_logit_fake)
                            ) )

D_solver    = tf.train.AdamOptimizer().minimize( D_loss, var_list = theta_D )
G_solver    = tf.train.AdamOptimizer().minimize( G_loss, var_list = theta_G )

sess    = tf.Session()
sess.run( tf.global_variables_initializer() )

if not os.path.exists( './out/' ):
    os.makedirs( './out/' )

# To save generated samples at each epoch
generated_n_samples = 7
Z_samp_gen   = sample_Z( generated_n_samples, Z_dim )

i = 0
epochs  = 100000
for it in range(epochs):
    if it%1000 == 0:
        fig = plt.figure( figsize = (generated_n_samples,10) )
        gs2  = gridspec.GridSpec(10,generated_n_samples)
        gs2.update( wspace = 0.05, hspace = 0.05 )
        generated   = []
        for k in range(10):    
            y_samples   = np.zeros( shape = [generated_n_samples,y_dim] )
            y_samples[:,k] = 1
            samples = sess.run( G_sample, feed_dict = {Z:Z_samp_gen,y:y_samples} )
            for j,s in enumerate(samples):
                ax  = plt.subplot( gs2[generated_n_samples*k+j] )
                plt.axis( 'off' )
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect( 'equal' )
                plt.imshow( s.reshape(28,28), cmap = 'Greys_r' )
        plt.savefig( './out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight' )
        i += 1
        plt.close( fig )

    X_mb, y_mb  = mnist.train.next_batch( mb_size )
    Z_samples   = sample_Z( mb_size, Z_dim )
    _, D_loss_curr  = sess.run( [D_solver,D_loss], feed_dict = {X:X_mb,Z:Z_samples,y:y_mb} )
    _, G_loss_curr  = sess.run( [G_solver,G_loss], feed_dict = {Z:Z_samples,y:y_mb} )
    if it%1000 == 0:
        print( "Epoch: {:8d}/{}    loss(D) = {:.9f}    loss(G) = {:.9f}".format(it,epochs,D_loss_curr,G_loss_curr) )
