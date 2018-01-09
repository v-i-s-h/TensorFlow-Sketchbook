# Gym

# Frozen Lake - Value Iteration

import gym
import numpy as np

game    = gym.make( 'FrozenLake-v0' )

transition_table = np.zeros( (game.observation_space.n,game.observation_space.n) )

# Run for some time and collect statistics
for i in range(1000):
    st  = game.reset()
    for stps in range(99):
        act =   game.action_space.sample()
        nst, rwd, dn, _ = game.step( act )
        transition_table[st,nst] += 1
        #
        if dn:
            print( "Epoch %d:    Done @ %3d with reward = %d" %(i,stps,rwd) )
            break

# Print
print( transition_table )