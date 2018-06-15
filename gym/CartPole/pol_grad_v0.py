# Policy Gradient Method
# From: https://github.com/awjuliani/DeepRL-Agents/blob/master/Policy-Network.ipynb

from __future__ import division

import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

env = gym.make( 'CartPole-v0' )

# Hyper-parameters
H           = 10            # No.of hidden layers
batch_size  = 5             # Every how many episodes to do gradient update
learning_rate   = 1e-2      # 
gamma       = 0.99          # Discount param
D           = 4             # Input dimension

tf.reset_default_graph()

# Make Network
observations    = tf.placeholder( tf.float32, [None,D], name = 'input_x' )
W1      = tf.get_variable( "W1", shape = [D,H], initializer = tf.contrib.layers.xavier_initializer() )
layer1  = tf.nn.relu( tf.matmul(observations,W1) )
W2      = tf.get_variable( "W2", shape = [H,1], initializer = tf.contrib.layers.xavier_initializer() )
score   = tf.matmul( layer1, W2 )
probability = tf.nn.sigmoid( score )

tvars   = tf.trainable_variables()
input_y = tf.placeholder( tf.float32, [None,1], name = 'input_y' )
advantages  = tf.placeholder( tf.float32, name = 'reward_signal' )

loglik  = tf.log( input_y*(input_y-probability) + (1-input_y)*(input_y+probability) )
loss    = -tf.reduce_mean( loglik * advantages )
newGrads    = tf.gradients( loss, tvars )

# To apply gradients
adam    = tf.train.AdamOptimizer( learning_rate = learning_rate )
W1Grad  = tf.placeholder( tf.float32, name = 'batch_grad1' )
W2Grad  = tf.placeholder( tf.float32, name = 'batch_grad2' )
batchGrad   = [ W1Grad, W2Grad ]
updateGrads = adam.apply_gradients( zip(batchGrad,tvars) )

# advantage function
def discount_rewards( r ):
    discounted_r    = np.zeros_like( r )
    running_add = 0
    for t in reversed(xrange(0,r.size)):
        running_add = gamma*running_add + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Training
xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward  = None
reward_sum      = 0
episode_number  = 1
total_episodes  = 10000

init    = tf.global_variables_initializer()


with tf.Session() as sess:
    rendering   = False
    sess.run( init )
    observation = env.reset()

    # Reset Gradient Placeholder
    gradBuffer  = sess.run( tvars )
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix]  = grad * 0

    while episode_number <= total_episodes:
        if reward_sum/batch_size > 100 or rendering == True:
            env.render()
            rendering   = True

        x   = np.reshape( observation, [1,D] )

        tfprob  = sess.run( probability, feed_dict = {observations:x} )
        action  = 1 if np.random.uniform() < tfprob else 0

        xs.append( x )
        y = 1 if action == 0 else 0 # Fake label??
        ys.append( y )

        observation, reward, done, info = env.step( action )
        reward_sum += reward

        drs.append( reward )

        if done:
            episode_number += 1

            epx = np.vstack( xs )
            epy = np.vstack( ys )
            epr = np.vstack( drs )
            tfp = tfps
            xs, ys, dlogps, drs, ys, tfps = [], [], [], [], [], []

            discounted_epr  = discount_rewards( epr )
            # Normalize
            discounted_epr  -= np.mean( discounted_epr )
            discounted_epr  //= np.std( discounted_epr )

            tGrad   = sess.run( newGrads, feed_dict = {
                                                        observations: epx,
                                                        input_y: epy,
                                                        advantages: discounted_epr
                                } )
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number%batch_size == 0:
                sess.run( updateGrads, feed_dict = {
                                                        W1Grad: gradBuffer[0],
                                                        W2Grad: gradBuffer[1]
                                                    } )
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix]  = grad * 0

                running_reward  = reward_sum if running_reward is None else running_reward*0.99+reward_sum*0.01

                print( "Average reward for episode {:7.3f}    Total Average Reward {:7.3f}".format(reward_sum//batch_size,
                                        running_reward//batch_size) )
                if reward_sum//batch_size > 200:
                    print( "Task solved in {} episodes!!".format(episode_number) )
                    break
                reward_sum = 0
            observation = env.reset()
