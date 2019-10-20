# Analyze model
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import larq as lq

import matplotlib.pyplot as plt

# Prepare dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Normalize pixel values between -1, and +1
train_images = train_images / 127.5 - 1.0
test_images = test_images / 127.5 - 1.0

model = load_model("./models/bnn_mnist.model")

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print("Test data   loss = {:.3f}    acc = {:.2f}".format(test_loss, test_acc))

model_input = model.input
model_output = [layer.output for layer in model.layers]
functor = K.function(model_input, model_output)

layer_outs = functor(test_images)
for (idx, out) in enumerate(layer_outs):
    print("{:3d}    {}".format(idx, out.shape))


fig, ax = plt.subplots(3, 1)

ax[0].hist(layer_outs[-1])
ax[1].hist(layer_outs[-2])
ax[2].hist(layer_outs[-3])

plt.show()