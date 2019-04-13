# Test for mixing Keras and TF for MLP

from __future__ import print_function

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

import tensorflow as tf

# Setup Custom Module Path
import os, sys
sys.path.append( './../' )

import Optimizers3 as Opt

import numpy as np

K.clear_session()
tf.reset_default_graph()

# ---------------- Create a dataset -------------------------
n_train = 1000
n_test = 100

y_train = np.random.randint(2, size=(n_train,2))
y_test = np.random.randint(2, size=(n_test,2))

x_train = 0.5*np.random.randn(n_train, 2) + y_train
x_test = 0.5*np.random.randn(n_test,2) + y_test

# --------------- Create model using Keras -------------------

with tf.name_scope("Model"):
  inputs = Input(shape=(2,))
  l1 = Dense(4, activation='relu')(inputs)
  l2 = Dense(4, activation='relu')(l1)
  predictions = Dense(2, activation='softmax')(l2)

model = Model(inputs=inputs, outputs=predictions)

model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer=SGD(),
#               metrics=['accuracy'])

# --------------- Create place holder and loss with TF------
with tf.name_scope("Model"):
  y_ph_tf = tf.placeholder(tf.float32, shape=[None, 2])
  loss = tf.losses.softmax_cross_entropy(y_ph_tf, predictions)

params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Model")

# trainer = tf.train.GradientDescentOptimizer(learning_rate=0.5)   # This works
trainer = Opt.SPSA(learning_rate=0.5)   # This don't
# _train = trainer.minimize(loss, var_list=params)
# _train = trainer.minimize(loss)
grads_and_vars = trainer.compute_gradients(loss)
_train = trainer.apply_gradients(grads_and_vars)

sess = K.get_session()

sess.run(tf.global_variables_initializer())

print(">>>>>> PRE")
print(sess.run(params[:2]))   # Print 

for i in range(20):
  l, _ = sess.run([loss,_train], feed_dict={inputs:x_train,y_ph_tf:y_train})
  print("{:4d}: {:f}".format(i, l))
  
print(">>>>>> POST")
print(sess.run(params[:2]))