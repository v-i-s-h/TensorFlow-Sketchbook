# Script to load a model HDF5 file and check entries

import os, argparse, json
import larq as lq
import tensorflow as tf

if __name__ == "__main__":

    input_shape = (28, 28, 1)
    num_classes = 10

    # To use binary weights
    kwargs = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
    )

    # # To use latent/real weights
    # kwargs = dict(
    #     input_quantizer="ste_sign",
    #     kernel_quantizer=None,
    #     kernel_constraint="weight_clip",
    # )
    

    # Create a simple model
    in_layer = tf.keras.layers.Input(shape=input_shape, name="m1_in")
    x = lq.layers.QuantConv2D(128, (3, 3),
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=input_shape,
                                name="m1_qc2d1")(in_layer)
    x = tf.keras.layers.BatchNormalization(scale=False, name="m1_bn1")(x)
    x = lq.layers.QuantConv2D(128, (3, 3), padding="same", **kwargs,
                                name="m1_qc2d2")(x)
    x = tf.keras.layers.BatchNormalization(scale=False, name="m1_bn2")(x)
    x = tf.keras.layers.Flatten(name="m1_f1")(x)
    x = lq.layers.QuantDense(num_classes, **kwargs,
                                name="m1_qd1")(x)
    x = tf.keras.layers.BatchNormalization(scale=False, name="m1_bn3")(x)
    x = tf.keras.layers.Dense(num_classes, name="m1_d1")(x)
    out_layer = tf.keras.layers.Activation("softmax", name="m1_out")(x)

    model1 = tf.keras.models.Model(inputs=in_layer, outputs=out_layer, name="Model1")

    # model1.summary()

    for l in model1.layers:
        print(l.name, [(w.name, w.shape) for w in l.trainable_weights])

    for w in model1.trainable_weights:
        print(w.name, w.shape)
