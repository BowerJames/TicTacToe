import random
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from ProbabilityDist import ProbabilityDist


class RandomAgent(object):

    def choose_action(self,state, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options-1)]


class IntelligentAgent(Model):

    def __init__(self, hidden_layer):
        super(IntelligentAgent, self).__init__()
        self.hidden_1 = Dense(hidden_layer, activation='relu')
        self.logits = Dense(9)

    def call(self, inputs):
        x = self.hidden_1(inputs)
        x = self.logits(x)
        return x

    def choose_action(self, state, possible_actions):
        fake_logits = self.call(np.expand_dims(state, axis=0))
        fake_logits_reshaped = tf.reshape(fake_logits, (9,))
        real_logits = tf.gather(fake_logits_reshaped, possible_actions)
        act = tf.squeeze(tf.random.categorical(tf.expand_dims(real_logits, 0), 1), axis=-1)
        return possible_actions[act]



