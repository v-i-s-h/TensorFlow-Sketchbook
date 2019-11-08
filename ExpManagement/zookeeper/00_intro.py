# Basic Example
# Source: https://github.com/larq/zookeeper
# With modifications from : https://github.com/plumerai/rethinking-bnn-optimization/blob/master/bnn_optimization/train.py

from os import path
import json
import click

import tensorflow as tf
from zookeeper import cli, build_train, HParams, registry, Preprocessing

# Callbacks
class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with open(path.join(path.dirname(self.filepath), "stats.json"), "w") as f:
            return json.dump({"epoch": epoch + 1}, f)

# Data preprocessing
class ImageClassification(Preprocessing):
    @property
    def kwargs(self):
        return {
            "input_shape": self.features["image"].shape,
            "num_classes": self.features["label"].num_classes
        }

    def inputs(self, data):
        return tf.cast(data["image"], tf.float32)

    def outputs(self, data):
        return tf.one_hot(data["label"], self.features["label"].num_classes)

@registry.register_preprocess("mnist")
class default(ImageClassification):
    def inputs(self, data):
        return super().inputs(data) / 255.0

@registry.register_preprocess("fashion_mnist")
class default(ImageClassification):
    def inputs(self, data):
        return super().inputs(data) / 255.0


@registry.register_preprocess("cifar10")
class default(ImageClassification):
    def inputs(self, data, training):
        image = data["image"]
        if training:
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)

        return tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0





@registry.register_model
def cnn(hp, input_shape, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            hp.filters[0], (3, 3), activation=hp.activation, input_shape=input_shape
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(hp.filters[1], (3, 3), activation=hp.activation),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(hp.filters[2], (3, 3), activation=hp.activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.filters[3], activation=hp.activation),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

@registry.register_hparams(cnn)
class basic(HParams):
    epochs = 10
    activation = "relu"
    batch_size = 32
    filters = [64, 64, 64, 64]
    learning_rate = 1e-3

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)


@registry.register_model
def fcn(hp, input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    for n in hp.nodes:
        x = tf.keras.layers.Dense(n, activation=hp.activation)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

@registry.register_hparams(fcn)
class basic(HParams):
    epochs = 10
    activation = "relu"
    batch_size = 32
    nodes = [128, 128, 64, 32]
    learning_rate = 1e-3

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)

@cli.command()
@click.option("--tensorboard/--no-tensorboard", default=True)
@build_train()
def train(build_model, dataset, hparams, output_dir, tensorboard):

    # Path to save model
    model_path = path.join(output_dir, "model")
    callbacks = [ ModelCheckpoint(filepath=model_path, save_weights_only=True) ]

    if tensorboard:
        callbacks.extend(
                [
                    tf.keras.callbacks.TensorBoard(
                        log_dir=output_dir,
                        write_graph=True,
                        histogram_freq=0,
                        # update_freq=250,
                        update_freq='epoch',
                        profile_batch=0,
                        embeddings_freq=0
                    ),
                ]
            )


    model = build_model(hparams, **dataset.preprocessing.kwargs)

    model.compile(
        optimizer=hparams.optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"]
    )

    model.fit(
        dataset.train_data(hparams.batch_size),
        epochs=hparams.epochs,
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=dataset.validation_data(hparams.batch_size),
        validation_steps=dataset.validation_examples // hparams.batch_size,
        callbacks=callbacks
    )

    # Save model
    model_name = build_model.__name__
    model.save_weights(path.join(output_dir, f"{model_name}_weights.h5"))

if __name__ == "__main__":
    cli()