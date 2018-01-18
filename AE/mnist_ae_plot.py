# Plot results from mnist_ae

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model, model_from_yaml
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.datasets import mnist

import matplotlib.pyplot as plt
import sys

encoding_dim    = 32

if len(sys.argv) < 2:
    print( "Error: No model file provided" )
    exit()

model_name  = sys.argv[1]
h_model_yaml    = open( model_name+".yaml", "r" )
autoencoder = model_from_yaml( h_model_yaml.read() )
h_model_yaml.close()
autoencoder.load_weights( model_name+".h5" )

print( "No.of layers: %d" % len(autoencoder.layers) )

# Encoder
encoder     = Model( autoencoder.layers[0].input, autoencoder.layers[1].output )
# # Decoder
encoded_input   = Input( shape=(encoding_dim,) )
decoder_layer   = autoencoder.layers[-1]
decoder     = Model( encoded_input, decoder_layer(encoded_input) )

autoencoder.compile( optimizer = 'adadelta', loss = 'binary_crossentropy' )

(x_train,_), (x_test,_) = mnist.load_data()
x_train = x_train.astype( 'float32' ) / 255
x_test  = x_test.astype( 'float32' ) / 255
x_train = x_train.reshape( (len(x_train),np.prod(x_train.shape[1:])) )
x_test  = x_test.reshape( (len(x_test),np.prod(x_test.shape[1:])) )

encoded_imgs    = encoder.predict( x_test )
decoded_imgs    = decoder.predict( encoded_imgs )

n = 25
plt.figure( figsize=(20,4) )
for i in range(n):
    ax  = plt.subplot( 2, n, i+1 )
    plt.imshow( x_test[i].reshape(28,28) )
    plt.gray()
    ax.get_xaxis().set_visible( False )
    ax.get_yaxis().set_visible( False )

    ax  = plt.subplot(2, n, i+1+n )
    plt.imshow( decoded_imgs[i].reshape(28,28) )
    plt.gray()
    ax.get_xaxis().set_visible( False )
    ax.get_yaxis().set_visible( False )
plt.show()