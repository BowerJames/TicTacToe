from tensorflow.keras.layers import Softmax
from tensorflow.keras import Model
import tensorflow as tf


class ProbabilityDist(Model):


    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)



