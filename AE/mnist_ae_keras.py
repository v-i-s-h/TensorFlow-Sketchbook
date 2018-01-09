# Keras implementation of Auto encoders for MNIST
# Based on: https://blog.keras.io/building-autoencoders-in-keras.html

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, model_from_yaml
from tensorflow.python.keras.datasets import mnist
import matplotlib.pyplot as plt
import datetime

logs_path   = "./mnist_ae_keras"

(x_train,_), (x_test,_) = mnist.load_data()
x_train = x_train.astype( 'float32' ) / 255
x_test  = x_test.astype( 'float32' ) / 255
x_train = x_train.reshape( (len(x_train),np.prod(x_train.shape[1:])) )
x_test  = x_test.reshape( (len(x_test),np.prod(x_test.shape[1:])) )

print x_train.shape
print x_test.shape

# Dimension to encode to
encoding_dim    = 32

input_img   = Input( shape=(784,) )
encoded     = Dense( encoding_dim, activation = 'relu' )( input_img )
decoded     = Dense( 784, activation = 'sigmoid' )( encoded )

autoencoder = Model( input_img, decoded )

# Encoder
encoder     = Model( input_img, encoded )
# Decoder
encoded_input   = Input( shape=(encoding_dim,) )
decoder_layer   = autoencoder.layers[-1]
decoder     = Model( encoded_input, decoder_layer(encoded_input) )

autoencoder.compile( optimizer = 'adadelta', loss = 'binary_crossentropy' )

autoencoder.fit( x_train, x_train,
                    epochs = 1,
                    batch_size = 256,
                    shuffle = True,
                    validation_data = (x_test,x_test) )

encoded_imgs    = encoder.predict( x_test )
decoded_imgs    = decoder.predict( encoded_imgs )

# Save model
model_name  = logs_path + "/ae" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
with open( model_name+".yaml", "w"  ) as model_yaml:
    model_yaml.write( autoencoder.to_yaml() )
autoencoder.save_weights( model_name+".h5" )
print "Model saved as '" + model_name + "'"



# n = 10
# plt.figure( figsize=(20,4) )
# for i in range(n):
#     ax  = plt.subplot( 2, n, i+1 )
#     plt.imshow( x_test[i].reshape(28,28) )
#     plt.gray()
#     ax.get_xaxis().set_visible( False )
#     ax.get_yaxis().set_visible( False )

#     ax  = plt.subplot(2, n, i+1+n )
#     plt.imshow( decoded_imgs[i].reshape(28,28) )
#     plt.gray()
#     ax.get_xaxis().set_visible( False )
#     ax.get_yaxis().set_visible( False )
# # plt.show()