# An attempt to make zookeeper scared
# Ex: TF_CPP_MIN_LOG_LEVEL=2 python main.py train fcn --dataset mnist --hparams activation="sigmoid"

from os import path
import click, json

import tensorflow as tf
from zookeeper import cli, build_train, HParams, registry, Preprocessing
from sacred import Experiment
from sacred.observers import MongoObserver

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

@registry.register_model
def fcn(hparams, input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    for num_of_nodes in hparams.nodes:
        x = tf.keras.layers.Dense(num_of_nodes, activation=hparams.activation)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

@registry.register_hparams(fcn)
class default(HParams):
    epochs = 5
    activation = "relu"
    batch_size = 32
    nodes = [256, 64]
    
    # Optimizer
    optimizer = "Adam"
    lr = 1e-3
    beta_1 = 0.99
    beta_2 = 0.999

    # @property
    # def opt(self):
    #     return tf.keras.optimizers.Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2)

@cli.command()
@click.option("--expname", default="Default Experiment")
@build_train()
def train(build_model, dataset, hparams, logdir, expname):

    # Location to save trained models
    output_dir = path.join(logdir, "test")

    # Create the actual train function to run
    def _train(_run):
        model = build_model(hparams, **dataset.preprocessing.kwargs)

        # Make optimizer
        optimizer = tf.keras.optimizers.Adam(hparams.lr, beta_1=hparams.beta_1, beta_2=hparams.beta_2)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )

        # model.summary()

        train_log = model.fit(
                        dataset.train_data(hparams.batch_size),
                        epochs=hparams.epochs,
                        steps_per_epoch=dataset.train_examples // hparams.batch_size,
                        validation_data=dataset.validation_data(hparams.batch_size),
                        validation_steps=dataset.validation_examples // hparams.batch_size
                    )
        
        # Log the performace values to sacred
        for (metric, values) in train_log.history.items():
            for (idx, value) in enumerate(values):
                _run.log_scalar(metric, value, idx)

        # TODO: Save model
        
    
    # Build config
    config = {}
    for (k, v) in hparams.items():
        config[k] = v
    # TODO: add model and dataset information also to config
    
    # Setup sacred experiment
    ex = Experiment(expname)
    my_url = '127.0.0.1:27018'  # Or <server-static-ip>:<port> if running on server
    ex.observers.append(MongoObserver.create(url=my_url,
                                         db_name='demo_data'))
    ex.main(_train)
    ex.add_config(config)


    # build argv for sacred -- hacky way!
    _argv = f"{ex.default_command}"
    ex.run_commandline(argv=_argv)

    click.echo("End of train")

if __name__ == "__main__":
    cli()
    # ex.run_commandline()