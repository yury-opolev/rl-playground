import tensorflow as tf
from keras import layers
from keras import models
from keras import initializers

class Model(object):
    def __init__(self, model_path, summary_path, checkpoint_dir, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_dir = checkpoint_dir

        self.nn_model = models.Sequential([
            layers.Input(shape=(20,)),
            layers.Dense(20, activation='relu',
                         kernel_initializer=initializers.RandomNormal(stddev=0.1),
                         bias_initializer=initializers.RandomNormal(stddev=0.1)),
            layers.Dense(20, activation='relu',
                         kernel_initializer=initializers.RandomNormal(stddev=0.1),
                         bias_initializer=initializers.RandomNormal(stddev=0.1)),
            layers.Dense(1, activation='sigmoid',
                         kernel_initializer=initializers.RandomNormal(stddev=0.1),
                         bias_initializer=initializers.RandomNormal(stddev=0.1))
        ])

        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.nn_model.load_weights(latest_checkpoint_path)

    def get_output(self, x):
        return self.nn_model(x)
