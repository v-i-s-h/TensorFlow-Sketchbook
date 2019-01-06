# Source: https://github.com/ardiya/siamesenetwork-tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

class Dataset( object ):
    def get_train_batch( self, n, onehot = True ):
        # Shuffle indices for random order every epoch
        np.random.shuffle( self.selTrainIdx )
        i = 0
        while i < self.nTrainSamples: 
            endIdx  = i+n if i < self.nTrainSamples else self.nTrainSamples
            idx = self.selTrainIdx[i:endIdx]
            if onehot:
                yield (self.trainX[idx],self.trainY1hot[idx])
            else:
                yield (self.trainX[idx],self.trainY[idx])
            i += n
    
    def get_test_batch( self, n, onehot = True ):
        # Shuffle indices for random order every epoch
        np.random.shuffle( self.selTestIdx )
        i = 0
        while i < self.nTestSamples: 
            endIdx  = i+n if i < self.nTestSamples else self.nTestSamples
            idx = self.selTestIdx[i:endIdx]
            if onehot:
                yield (self.testX[idx],self.testY1hot[idx])
            else:
                yield (self.testX[idx],self.testY[idx])
            i += n

    def get_other_batch( self, n, onehot = True ):
        # Shuffle indices for random order every epoch
        np.random.shuffle( self.otherTrainIdx )
        i = 0
        while i < self.nOthSamples: 
            endIdx  = i+n if i < self.nOthSamples else self.nOthSamples
            idx = self.otherTrainIdx[i:endIdx]
            if onehot:
                yield (self.trainX[idx],self.trainY1hot[idx])
            else:
                yield (self.trainX[idx],self.trainY[idx])
            i += n
    

class MNISTDataset( Dataset ):
    def __init__( self, nClasses = 7 ):
        print( "===== Loading MNIST Dataset =====" )
        self.nClasses = nClasses
        (trainX, trainY), (testX,testY) = mnist.load_data()
        self.trainX = np.expand_dims(trainX,axis=3) / 255.0
        self.testX  = np.expand_dims(testX,axis=3) / 255.0
        self.trainY = np.expand_dims( trainY, axis = 1 )
        self.testY  = np.expand_dims( testY, axis = 1 )

        self.uniqueLabels    = np.unique( self.trainY )
        trainIndices    = { label: np.flatnonzero(trainY==label) for label in self.uniqueLabels }
        testIndices     = { label: np.flatnonzero(testY==label) for label in self.uniqueLabels }

        self.selectedLabels = np.sort( np.random.choice( self.uniqueLabels, nClasses, replace = False  ) )
        self.otherLabels    = np.setdiff1d( self.uniqueLabels, self.selectedLabels, assume_unique = True )
        self.selTrainIdx    = np.concatenate( [trainIndices[i] for i in self.selectedLabels] )
        self.selTestIdx     = np.concatenate( [testIndices[i] for i in self.selectedLabels] )
        self.otherTrainIdx  = np.concatenate( [trainIndices[i] for i in self.otherLabels] )
        self.otherTestIdx   = np.concatenate( [testIndices[i] for i in self.otherLabels] )

        self.nTrainSamples  = self.selTrainIdx.shape[0]
        self.nTestSamples   = self.selTestIdx.shape[0]
        self.nOthSamples    = self.otherTrainIdx.shape[0]

        # Make one-hot
        self.idx2label  = { idx: label for (idx,label) in enumerate(self.selectedLabels) }
        self.label2idx  = { label: idx for (idx,label) in self.idx2label.items() }
        self.trainY1hot = np.zeros( (self.trainX.shape[0],self.nClasses) )
        self.testY1hot  = np.zeros( (self.testY.shape[0],self.nClasses) )
        for i in range(self.trainY.shape[0]):
            thisLabel = self.trainY[i,0]
            if thisLabel in self.selectedLabels:
                thisIdx = self.label2idx[thisLabel]
                self.trainY1hot[i,thisIdx] = 1.0
        for i in range(self.testY.shape[0]):
            thisLabel = self.testY[i,0]
            if thisLabel in self.selectedLabels:
                thisIdx = self.label2idx[thisLabel]
                self.testY1hot[i,thisIdx] = 1.0
            
        
        print( "sel: ", self.selectedLabels )
        print( "oth: ", self.otherLabels )
        print( "selTrainIdx : ", self.selTrainIdx.shape )
        print( "selTestIdx  : ", self.selTestIdx.shape )
        print( "othTrainIdx : ", self.otherTrainIdx.shape )
        print( "othTestIdx  : ", self.otherTestIdx.shape )
        print( "label2idx   : ", self.label2idx )
        print( "idx2label   : ", self.idx2label )  
        print( "nTrainSample: ", self.nTrainSamples )
        print( "nTestSamples: ", self.nTestSamples)

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    a = MNISTDataset( 6 )

    trainGen = a.get_other_batch( 4*4, onehot = True )
    (x,y) = next( trainGen )
    
    f, a = plt.subplots( 4, 4 )
    for (i,ax1) in enumerate(a):
        for (j,ax2) in enumerate(ax1):
            ax2.imshow( np.squeeze(x[i*4+j,],axis=2) )
    print( y )
    plt.show()
    