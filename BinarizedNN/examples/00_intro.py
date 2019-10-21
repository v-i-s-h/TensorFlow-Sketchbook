# Source: https://larq.dev/examples/mnist/

import tensorflow as tf
import larq as lq

# Prepare dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Normalize pixel values between -1, and +1
train_images = train_images / 127.5 - 1.0
test_images = test_images / 127.5 - 1.0

# All layers except first layer will use the same option
kwargs = dict(input_quantizer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip")

model = tf.keras.Sequential()

# In the first layer, we only quantize the weights, not the weights
model.add(lq.layers.QuantConv2D(32, (3, 3), kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=(28, 28, 1)))

# Create model
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.LayerNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.LayerNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
# model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.LayerNormalization(scale=False))
model.add(tf.keras.layers.Flatten())

model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
# model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.LayerNormalization(scale=False))

model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
# model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.LayerNormalization(scale=False))
model.add(tf.keras.layers.Activation("softmax"))

# Summary
lq.models.summary(model)

# Get ready for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=64, epochs=6)

# Test 
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print("Test data   loss = {:.3f}    acc = {:.2f}".format(test_loss, test_acc))

model.save("./models/bnn_mnist_ln.model")