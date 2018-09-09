
import tensorflow as tf

from tensorflow.contrib.opt import PowerSignOptimizer as PSO
import numpy as np

# Setup Custom Module Path
import os, sys
sys.path.append( './../' )
import Optimizers

import matplotlib.pyplot as plt


def RosenbrockOpt( optimizer, MAX_EPOCHS = 4000, MAX_STEP = 100 ):
    x1_data = tf.Variable( initial_value = tf.random_uniform([1],minval=-3,maxval=+3,seed=0), name = "x1" )
    x2_data = tf.Variable( initial_value = tf.random_uniform([1],minval=-3,maxval=+3,seed=1), name = "x2" )

    y = tf.add( 
        tf.pow( tf.subtract(1.0,x1_data), 2.0 ),
        tf.multiply( 100.0,  tf.pow(tf.subtract(x2_data,tf.pow(x1_data,2.0)),2.0)),
        name = 'y'
    )

    global_step_tensor  = tf.Variable( 0, trainable = False, name = "global_step" )

    train   = optimizer.minimize( y, global_step = global_step_tensor )

    sess    = tf.Session()

    init    = tf.global_variables_initializer()
    sess.run( init )

    minx    = 1.0
    miny    = 1.0

    distance    = []
    xx_ = sess.run( x1_data )
    yy_ = sess.run( x2_data )
    print( "Step: {:5d}   x_1 = {:5.4f}   x_2 = {:5.4f}   y = {:5.4f}".format(0,xx_[0],yy_[0],np.sqrt((minx-xx_)**2+(miny-yy_)**2)[0]) )
    for step in range( MAX_EPOCHS ):
        _, xx_, yy_, zz_    = sess.run( [train,x1_data,x2_data,y] )
        if step%MAX_STEP == 0:
            print( "Step: {:5d}   x_1 = {:5.4f}   x_2 = {:5.4f}   y = {:5.4f}".format(step+1, xx_[0],yy_[0], zz_[0]) )
            distance += [ np.sqrt((minx-xx_)**2+(miny-yy_)**2)[0] ]
        
    sess.close()

    return distance

list_of_opt = {
    # "SGD(0.0007)": tf.train.GradientDescentOptimizer(0.0007),
    "SGD(0.0006)": tf.train.GradientDescentOptimizer(0.0006),
    "PowerSign": Optimizers.PowerSignOptimizer(),
    "PSO": PSO( learning_rate = 0.000025, base = 0.05, beta = 50 )
}

data = []

for opt in list_of_opt.keys():
    dist    = RosenbrockOpt( list_of_opt[opt] )
    data.append( {
        "label": opt,
        "data": dist
    } )

fig = plt.figure()
for d in data:
    plt.plot( d["data"], label = d["label"] )
plt.grid( True )
plt.legend()
plt.show()