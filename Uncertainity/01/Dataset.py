# Source: https://github.com/ardiya/siamesenetwork-tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

class Dataset( object ):
    def get_train_batch( self, n ):
        idx = np.random.choice( self.selTrainIdx, size = n , replace = False )
        return self.trainX[idx], self.trainY[idx]
    
    def get_test_batch( self, n ):
        idx = np.random.choice( self.selTestIdx, size = n, replace = False )
        return self.testX[idx], self.testY[idx]

    def get_other_batch( self, n ):
        idx = np.random.choice( self.otherTrainIdx, size = n, replace = False )
        return self.trainX[idx], self.trainY[idx]
    

class MNISTDataset( Dataset ):
    def __init__( self, nClasses = 7 ):
        print( "===== Loading MNIST Dataset =====" )
        (trainX, trainY), (testX,testY) = mnist.load_data()
        self.trainX = np.expand_dims(trainX,axis=3) / 255.0
        self.testX  = np.expand_dims(testX,axis=3) / 255.0
        self.trainY = np.expand_dims( trainY, axis = 1 )
        self.testY  = np.expand_dims( testY, axis = 1 )

        self.uniqueLabels    = np.unique( self.trainY )
        trainIndices    = { label: np.flatnonzero(trainY==label) for label in self.uniqueLabels }
        testIndices     = { label: np.flatnonzero(testY==label) for label in self.uniqueLabels }
        
        self.selectedLabels = np.random.choice( self.uniqueLabels, nClasses, replace = False )
        self.otherLabels    = np.setdiff1d( self.uniqueLabels, self.selectedLabels, assume_unique = True )
        print( "sel: ", self.selectedLabels )
        print( "oth: ", self.otherLabels )
        self.selTrainIdx    = np.concatenate( [trainIndices[i] for i in self.selectedLabels] )
        self.selTestIdx     = np.concatenate( [testIndices[i] for i in self.selectedLabels] )
        self.otherTrainIdx  = np.concatenate( [trainIndices[i] for i in self.otherLabels] )
        self.otherTestIdx   = np.concatenate( [testIndices[i] for i in self.otherLabels] )
        
        print( "selTrainIdx : ", self.selTrainIdx.shape )
        print( "selTestIdx  : ", self.selTestIdx.shape )
        print( "othTrainIdx : ", self.otherTrainIdx.shape )
        print( "othTestIdx  : ", self.otherTestIdx.shape )
        
if __name__ == "__main__":
    a = MNISTDataset( 6 )