# A implementation of DQN for CartPOle Environment
# Based on: https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/

import random
import gym
import numpy as np
import math

from collections import deque
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam

class DQNSolver():
    def __init__( self, 
                    n_episodes = 2500, 
                    n_win_ticks = 195, 
                    max_env_ticks = None, 
                    gamma = 1.0, 
                    epsilon = 1.0, 
                    epsilon_min = 0.01, 
                    epsilon_log_decay = 0.995, 
                    alpha = 0.01, 
                    alpha_decay = 0.01, 
                    batch_size = 64, 
                    monitor = False, 
                    quiet = False ):
        # For memory Replay
        self.memory = deque( maxlen = 10000 )
        # Make Environment
        self.env    = gym.make( 'CartPole-v0' )
        # Setup algorithm parameters
        self.n_episodes     = n_episodes
        self.n_win_ticks    = n_win_ticks
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_log_decay
        self.alpha          = alpha
        self.alpha_decay    = alpha_decay
        self.batch_size     = batch_size
        self.quiet          = quiet
        if max_env_ticks is not None:
            self.env._max_episode_steps = max_env_ticks

        # Make ANN model
        self.model  = Sequential()
        self.model.add( Dense( 10, input_dim = 4, activation = 'tanh' ) )
        self.model.add( Dense( 10, activation = 'tanh' ) )
        self.model.add( Dense( 2, activation = 'linear' ) )
        self.model.compile( loss = 'mse', optimizer = Adam( lr = self.alpha, decay = self.alpha_decay ) )

    # Save for memory replay
    def remember( self, state, action, next_state, reward, done ):
        self.memory.append( (state,action,reward,next_state,done) )
    
    # To choose an action
    def choose_action( self, state, epsilon ):
        return self.env.action_space.sample() if ( np.random.random() <= epsilon ) else np.argmax( self.model.predict(state) )

    def get_epsilon( self, t ):
        return max( self.epsilon_min, min(self.epsilon,1.0-math.log10((t+1)*self.epsilon_decay)) )

    def preprocess_state( self, state ):
        return np.reshape( state, [1,4] )

    def replay( self, batch_size ):
        x_batch, y_batch = [], []
        # print( "My Memory = {}".format(len(self.memory)) )
        minibatch = random.sample( self.memory, min(len(self.memory),batch_size) )
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict( state )
            y_target[0][action] = reward if done else reward + self.gamma*np.max(self.model.predict(next_state)[0])
            x_batch.append( state[0] )
            y_batch.append( y_target[0] )
        self.model.fit( np.array(x_batch), np.array(y_batch), batch_size = len(x_batch), verbose = 0 )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run( self ):
        scores = deque( maxlen = 100 )

        for e in range(self.n_episodes):
            state = self.preprocess_state( self.env.reset() )
            done  = False
            i     = 0
            while not done:
                action      = self.choose_action( state, self.get_epsilon(e) )
                next_state, reward, done, _ = self.env.step( action )
                next_state  = self.preprocess_state( next_state )
                self.remember( state, action, next_state, reward, done )
                state = next_state
                i += 1
                if e%100 == 0:
                    self.env.render()

            scores.append( i )
            mean_score = np.mean( scores )
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet:
                    print( "Ran {} episodes. Solved after {} trials".format(e,e-100) )
                    # return e-100
            if e%100 == 0  and not self.quiet:
                print( "Episode {}: Mean survival time in last 100 episodes = {}".format(e,mean_score) )
                print( "State = {}    Prediction = {}".format(state,self.model.predict(state)) )
            self.replay( self.batch_size )

        if not self.quiet:
            print( "Did not solve after {} episodes".format(e) )
        return e

if __name__ == '__main__':
    agent = DQNSolver()
    agent.run()