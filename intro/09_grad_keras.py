# To compute gradients for L-layer network

import tensorflow as tensorflow
import tensorflow.keras as keras
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.datasets import mnist

import numpy as np

N_HLAYERS = 6
N_NEURONS = 4

# ------------------------------- Prepare dataset ------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test =x_test.reshape(-1, 28*28) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]
# ------------------------------------------------------------------------------

inputs = Input(shape=(784,))

x = inputs
for i in range(N_HLAYERS):
    x = Dense(N_NEURONS, activation='tanh')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='SGD', loss='categorical_crossentropy')

model.summary()


# For obtaining gradients
sess = K.get_session()
gradients = K.gradients(model.output, model.trainable_weights)

grad_val = sess.run(gradients, feed_dict={model.input: x_train[0:1,:]})


grads_dict = {}

for idx in range(N_HLAYERS+1):
    grads_dict["W_{:03d}".format(idx)] = grad_val[2*idx+0]
    grads_dict["b_{:03d}".format(idx)] = grad_val[2*idx+1]

print("Gradients")
for (k,v) in grads_dict.items():
    print(k, v.shape)