# DQN from: https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial5/dqn-cartpole.py

import gym
import random
import numpy as np

import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.regularizers import l2

# Memory Buffer
class Memory:
    # (s,a,r,s') -> Dictionary( 'state', 'action', 'reward', 'nextState', 'isFinal' )
    def __init__( self, size ):
        self.size   = size
        self.currentPosition    = 0
        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.nextStates = []
        self.finals     = []

    def getMiniBatch( self, size ):
        indices = random.sample( np.arange(len(self.states)), min(size,len(self.states)) )
        miniBatch   = []
        for index in indices:
            miniBatch.append( {
                'state'     : self.states[index],
                'action'    : self.actions[index],
                'reward'    : self.rewards[index],
                'nextState' : self.nextStates[index],
                'isFinal'   : self.finals[index]
            } )
        return miniBatch

    def getCurrentSize( self ):
        return len( self.states )
    
    def getMemory( self, index ):
        return {
            'state'     : self.states[index],
            'action'    : self.actions[index],
            'reward'    : self.rewards[index],
            'nextState' : self.nextStates[index],
            'isFinal'   : self.finals[index]
        }

    def addMemory( self, state, action, nextState, reward, isFinal ):
        if self.currentPosition >= self.size-1:
            self.currentPosition    = 0             # For round robin memory
        if len(self.states) > self.size:
            self.states[self.currentPosition]   = state
            self.actions[self.currentPosition]  = action
            self.nextStates[self.currentPosition]    = nextState
            self.rewards[self.currentPosition]  = reward
            self.finals[self.currentPosition]   = isFinal
        else:
            self.states.append( state )
            self.actions.append( action )
            self.nextStates.append( nextState )
            self.rewards.append( reward )
            self.finals.append( isFinal )
        self.currentPosition    += 1

class DeepQ:
    def __init__( self, 
                    inputs,             # Input size
                    outputs,            # Output size
                    memorySize,         # Memory Size
                    discountFactor,     # Discount Factor for Q-learning
                    learningRate,       # learning rate
                    learnStart      ):  # Steps to happen before learning
        self.input_size     = inputs
        self.output_size    = outputs
        self.memory         = Memory( memorySize )
        self.discountFactor = discountFactor
        self.learningRate   = learningRate
        self.learnStart     = learnStart

    def initNetworks( self, hiddenLayers ):
        model   = self.createModel( self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate )
        self.model  = model

        tagertModel = self.createModel( self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate )
        self.tagertModel    = tagertModel

    def createRegularizedModel( self, 
                                inputs, 
                                outputs, 
                                hiddenLayers,       # List of nodes at each hidden layer
                                activationType, 
                                learningRate ):
        bias    = True
        dropout = 0
        regularizationFactor    = 0.01
        model   = Sequential()
        if len(hiddenLayers) == 0:
            model.add( Dense(self.output_size,
                                input_shape=(self.input_size,),
                                init='lecun_uniform',bias=bias) )
            model.add( Activation("linear") )
        else:
            if regularizationFactor > 0:
                model.add( Dense(hiddenLayers[0],
                                    input_shape=(self.input_size,),
                                    init='lecun_uniform',
                                    W_regularizer=l2(regularizationFactor),
                                    bias=bias) )
            else:
                model.add( Dense(hiddenLayers[0],
                                    input_shape=(self.input_size,),
                                    init='lecun_uniform',
                                    bias=bias) )
            if activationType == 'LeakyReLU':
                model.add( LeakyReLU(alpha=0.01) )
            else:
                model.add( Activation(activationType) )

            for index in range(1,len(hiddenLayers)):
                layerSize   = hiddenSize[index]
                if regularizationFactor > 0.0:
                    model.add( Dense(hiddenLayers[index],
                                        init='lecun_uniform',
                                        W_regularizer=l2(regularizationFactor),
                                        bias=bias) )
                else:
                    model.add( Dense(hiddenLayers[index],
                                    init='lecun_uniform',
                                    bias=bias) )
                if activationType == "LeakyReLU":
                    model.add( LeakyReLU(alpha=0.01) )
                else:
                    model.add( Activation(activationType) )
                if dropout > 0:
                    model.add( Dropout(dropout) )
            model.add( Dense(self.output_size,
                                init='lecun_uniform',
                                bias=bias) )
            model.add( Activation("linear") )
        optimizer   = optimizers.RMSprop( lr = learningRate, rho = 0.9, epsilon = 1e-6 )
        model.compile( loss = "mse", optimizer = optimizer )
        model.summary()
        return model

    def createModel( self, inputs, outputs, hiddenLayers, activationType, learningRate ):
        model   = Sequential()
        if len(hiddenLayers) == 0:
            model.add( Dense(self.output_size,
                                input_shape=(self.input_size,),
                                init='lecun_uniform') )
            model.add( Activation('linear') )
        else:
            model.add( Dense(hiddenLayers[0],
                                input_shape=(self.input_size,),
                                kernel_initializer='lecun_uniform') ) 
            if activationType == 'LeakyReLU':
                model.add( LeakyReLU(alpha=0.01) )
            else:
                model.add( Activation(activationType) )

            for index in range(1,len(hiddenLayers)):
                layerSize   = hiddenLayers[index]
                model.add( Dense(layerSize,kernel_initializer='lecun_uniform') )
                if activationType == 'LeakyReLU':
                    model.add( LeakyReLU(alpha=0.01) )
                else:
                    model.add( Activation(activationType) )
            model.add( Dense(self.output_size,kernel_initializer='lecun_uniform') )
            model.add( Activation('linear') )
        optimizer   = optimizers.RMSprop( lr = learningRate, rho = 0.9, epsilon = 1e-6 )
        model.compile( loss = "mse", optimizer = optimizer )
        model.summary()
        return model

    def printNetwork( self ):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print( "Layer#{:02d}: {}".format(i,weights) )
            i += 1
        
    def backupNetwork( self, model, backup ):
        weightMatrix    = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append( weights )
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights( weights )
            i += 1
    
    def updateTargetNetwork( self ):
        self.backupNetwork( self.model, self.tagertModel )

    def getQValues( self, state ):
        predicted   = self.model.predict( state.reshape(1,len(state)) )
        return predicted[0]
    
    def getTargetQValues( self, state ):
        predicted   = self.model.predict( state.reshape(1,len(state)) )
        return predicted[0]

    def getMaxQ( self, qValues ):
        return np.max( qValues )
    
    def getMaxIndex( self, qValues ):
        return np.argmax( qValues )

    def calculateTarget( self, qValuesNewState, reward, isFinal ):
        if isFinal:
            return reward
        else:
            return reward + self.discountFactor*self.getMaxQ(qValuesNewState)
        
    def selectAction( self, qValues, explorationRate ):
        rand    = random.random()
        if rand < explorationRate:
            action  = np.random.randint( 0, self.output_size )
        else:
            action  = self.getMaxIndex( qValues )
        return action

    def selectActionByProbability( self, qValues, bias ):
        qValueSum   = 0.0
        shiftBy     = 0.0
        for value in qValues:
            if value+shiftBy < 0:
                shiftBy = -(value+shiftBy)
        shiftBy += 1e-6

        for value in qValues:
            qValueSum += (value+shiftBy)**bias
        
        probabilitySum  = 0.0
        qValuesProbabilities    = []
        for value in qValues:
            probability = ((value+shiftBy)**bias) / float(qValueSum)
            qValuesProbabilities.append( probability + probabilitySum )
            probabilitySum += probability
        qValuesProbabilities[len(qValuesProbabilities)-1] = 1.0

        rand    = random.random()
        i   = 0
        for value in qValuesProbabilities:
            if rand <= value:
                return i
            i += 1
        # end for
    # end def    

    def addMemory( self, state, action, nextState, reward, isFinal ):
        self.memory.addMemory( state, action, nextState, reward, isFinal )
    
    def learnOnLastState( self ):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory( self.memory.getCurrentSize()-1 )
    
    def learnOnMiniBatch( self, miniBatchSize, useTargetNetwork = True ):
        if self.memory.getCurrentSize() > self.learnStart:
            miniBatch   = self.memory.getMiniBatch( miniBatchSize )
            X_batch     = np.empty( (0,self.input_size), dtype = np.float64 )
            Y_batch     = np.empty( (0,self.output_size), dtype = np.float64 )
            for sample in miniBatch:
                isFinal     = sample['isFinal']
                state       = sample['state']
                action      = sample['action']
                nextState   = sample['nextState']
                reward      = sample['reward']

                qValues     = self.getQValues( state )
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues( nextState )
                else:
                    qValuesNewState = self.getQValues( nextState )
                targetValue = self.calculateTarget( qValuesNewState, reward, isFinal )

                X_batch = np.append( X_batch, np.array([state.copy()]), axis = 0 )
                Y_sample    = qValues.copy()
                Y_sample[action]    = targetValue
                Y_batch = np.append( Y_batch, np.array([Y_sample]), axis = 0 )
                if isFinal:
                    X_batch = np.append( X_batch, np.array([nextState.copy()]), axis = 0 )
                    Y_batch = np.append( Y_batch, np.array([[reward]*self.output_size]), axis = 0)
            self.model.fit( X_batch, Y_batch, batch_size = len(miniBatch), epochs = 1, verbose = 0 )
    
env = gym.make( 'CartPole-v0' )
    
epochs  = 1000
steps   = 100000
updateTargetNetwork = 10000
explorationRate = 1.00
minibatch_size  = 128
learnStart      = 128
learningRate    = 0.00025
discountFactor  = 0.99
memorySize      = 1000000

last100Scores       = [0] * 100
last100ScoresIndx   = 0
last100Filled       = False

deepQ   = DeepQ( 4, 2, memorySize, discountFactor, learningRate, learnStart )
deepQ.initNetworks( [300,300] )

stepCounter = 0

for epoch in xrange(epochs):
    observation = env.reset()
    print( "Epoch#{:05d}".format(epoch) )
    print( "    Exploration Rate = {:5f}".format(explorationRate) )
    for t in xrange(steps):
        qValues = deepQ.getQValues( observation )
        action  = deepQ.selectAction( qValues, explorationRate )
        nextState, reward, done, info   = env.step( action )

        if t >= 199:
            print( "    **** Reached the end!! :D ****" )
            done    = True
        if done and t < 199:
            print( "    [Decrease reward]    " )

        deepQ.addMemory( observation, action, nextState, reward, done )

        if stepCounter >= learnStart:
            if stepCounter <= updateTargetNetwork:
                deepQ.learnOnMiniBatch( minibatch_size, False )
            else:
                deepQ.learnOnMiniBatch( minibatch_size, True )
        
        observation = nextState

        if done:
            last100Scores[last100ScoresIndx]    = t
            last100ScoresIndx   += 1
            if last100ScoresIndx >= 100:
                last100Filled       = True
                last100ScoresIndx   = 0
            if not last100Filled:
                print( "    Finished after {:5d} timesteps".format(t+1) )
            else:
                print( "    Finished after {:5d} timesteps".format(t+1) )
                print( "    Last 100 average: {:.4f}".format((sum(last100Scores)/len(last100Scores))) )
            break

        stepCounter += 1
        if stepCounter%updateTargetNetwork  == 0:
            deepQ.updateTargetNetwork()
            print( "---------------------Updating Target Network...----------------------" )
    
    explorationRate *= 0.995
    explorationRate = max( 0.05, explorationRate )