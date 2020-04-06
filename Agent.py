import copy
import random
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from ProbabilityDist import ProbabilityDist


class RandomAgent(object):

    def choose_action(self, state, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options-1)]

    def choose_top_action(self, state, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]


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
        return possible_actions[act[0]]

    def choose_top_action(self, state, possible_actions):
        fake_logits = self.call(np.expand_dims(state, axis=0))
        fake_logits_reshaped = tf.reshape(fake_logits, (9,))
        real_logits = tf.gather(fake_logits_reshaped, possible_actions)
        act = tf.math.argmax(real_logits)
        return possible_actions[act]

    def update_actor(self, states, actions, rewards, learning_rate=0.001):
        one_hot_actions = np.zeros((actions.size, 9))
        rows = np.arange(actions.size)
        one_hot_actions[rows, actions.astype(int)] = 1

        with tf.GradientTape() as tape:
            logits = self.call(states)
            log_inp = tf.boolean_mask(logits, one_hot_actions)
            loss = tf.losses.categorical_crossentropy(rewards, log_inp, from_logits=True)
            reg = tf.nn.l2_loss(logits)

            loss += 0.1*reg
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.SGD(learning_rate)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def update_agent_2(self, states, actions, rewards, possible_actions_func, learning_rate=0.01):
        updated_agent = copy.deepcopy(self)
        number_states = len(actions)
        for i in range(number_states):
            state = states[i, :]
            action = actions.astype(int)[i]
            reward = rewards[i]
            with tf.GradientTape() as tape:
                x = self.call(np.expand_dims(state,0))
                possible_actions = possible_actions_func(state)
                x = tf.reshape(x, (9,))
                x = tf.gather(x, possible_actions)
                x = tf.nn.softmax(x)
                x = tf.math.log(x)
                x = tf.gather(x, np.where(possible_actions == action))
                x = -1*reward*x
            grads = tape.gradient(x, self.trainable_variables)
            optimizer = tf.keras.optimizers.SGD(learning_rate/number_states)
            optimizer.apply_gradients(zip(grads, updated_agent.trainable_variables))

        return updated_agent










