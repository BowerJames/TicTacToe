import random
import numpy as np
import os
import ConnectMemory
from tensorflow.keras.models import load_model, clone_model
import tensorflow.keras.optimizers as opt
import tensorflow as tf
import Connect4

class ConnectAgent(object):

    def __init__(self, qnet=None, memory=None):
        self.qnet = qnet
        self.memory = memory

    def choose_random_action(self, state, possible_actions_func):
        possible_actions = possible_actions_func(state)
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def choose_greedy_action(self, state, possible_action_func):
        top_action, _ = self.top_action(state, possible_action_func)
        return top_action

    def choose_eps_greedy_action(self, state, possible_action_func, eps=0.3):
        if random.random() < eps:
            return self.choose_random_action(state, possible_action_func)
        else:
            return self.choose_greedy_action(state, possible_action_func)

    def top_action(self, state, possible_action_func):
        top_action = None
        top_action_value = None
        for action in possible_action_func(state):
            one_hot_action = np.zeros((1, 6, 7, 1))
            one_hot_action[0, :, action, 0] = 1
            state_action = np.concatenate((np.expand_dims(state, axis=0), one_hot_action), axis=3)
            action_value = self.qnet.predict_on_batch(state_action)
            if top_action is None:
                top_action = action
                top_action_value = action_value
            elif action_value > top_action_value:
                top_action = action
                top_action_value = action_value

        return top_action, top_action_value

    def update_memory(self, sars):
        self.memory.push(sars)

    def memory_full(self):
        return self.memory.isfull()

    def clear_memory(self):
        self.memory.empty()

    def train(self, number_states, batch_size, possible_actions_func, epochs=1, discount_factor=-1):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(number_states)
        state_batch = np.asarray(state_batch)
        action_batch = np.asarray(action_batch)
        reward_batch = np.asarray(reward_batch)
        next_state_batch = np.asarray(next_state_batch)
        done_batch = np.asarray(done_batch)

        state_actions = np.concatenate((state_batch, action_batch), axis=3)
        targets = reward_batch + (1-done_batch) * discount_factor * self.top_action_values(next_state_batch, possible_actions_func)

        '''log_dir = '.\Tensorboard'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)'''

        self.qnet.compile(optimizer=opt.Adam(learning_rate=0.00001), loss='mse')
        self.qnet.fit(state_actions, targets, batch_size=batch_size, epochs=epochs)

    def top_action_values(self, states, possible_actions_func):
        top_action_values = np.empty((0,))
        for state in states:
            if state is None:
                top_action_values = np.append(top_action_values, 0)
            else:
                top_action_values = np.append(top_action_values, self.top_action(state, possible_actions_func)[1])
        return top_action_values

    def memory_size(self):
        return len(self.memory)


    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.qnet.save(dir_path + '/qnet.h5')
        self.memory.save(dir_path + '/memory.pickle')

    @staticmethod
    def load(file_path):
        net = load_model(file_path + '/qnet.h5')
        memory = ConnectMemory.ValueFunctionMemory.load(file_path + '/memory.pickle')
        return ConnectAgent(qnet=net, memory=memory)

class StateToActionAgent(object):
    def __init__(self, qnet=None, target_net = None, memory=None):
        self.qnet = qnet
        self.memory = memory
        self.target_net = target_net

    def choose_random_action(self, state, possible_actions_func):
        possible_actions = possible_actions_func(state)
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def choose_greedy_action(self, state, possible_action_func):
        top_action, _ = self.top_action(state, possible_action_func)
        return top_action

    def update_target_weights(self):
        self.target_net.set_weights(self.qnet.get_weights())

    def top_action(self, state, possible_actions_func):
        action_values = self.action_values_live(tf.convert_to_tensor(np.expand_dims(state, axis=0)))
        top_action = None
        top_action_value = None
        for action in possible_actions_func(state):
            if top_action is None:
                top_action = action
                top_action_value = action_values[0, action]
            elif action_values[0, action] > top_action_value:
                top_action = action
                top_action_value = action_values[0, action]
        return top_action, top_action_value

    def choose_eps_greedy_action(self, state, possible_action_func, eps=0.3):
        if random.random() < eps:
            return self.choose_random_action(state, possible_action_func)
        else:
            return self.choose_greedy_action(state, possible_action_func)

    @tf.function
    def action_values_live(self, state):
        return self.qnet(state)


    def train(self, num_batches, batch_size, possible_actions_func, discount_factor=-1):
        if self.memory_size() < batch_size:
            return
        for i in range(num_batches):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)
            state_batch = np.array(state_batch)
            belief = self.qnet.predict_on_batch(state_batch).numpy()
            for j in range(batch_size):
                belief[j, action_batch[j]] = reward_batch[j] if done_batch[j] else reward_batch[j] + discount_factor * self.top_action(next_state_batch[j], possible_actions_func)[1].numpy()
            '''if loss is None:
                loss = self.qnet.evaluate(x=state_batch, y=belief)'''

            self.qnet.train_on_batch(state_batch, belief)

    def train2(self, num_batches, optimizer, batch_size, possible_actions_func, discount_factor=-1):
        if self.memory_size() < batch_size:
            return
        for i in range(num_batches):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)
            state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float64)
            action_batch = tf.convert_to_tensor(action_batch)
            reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float64)
            next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float64)
            done_batch = tf.convert_to_tensor(done_batch)

            '''if loss is None:
                loss = self.qnet.evaluate(x=state_batch, y=belief)'''

            self.update_batch(state_batch, action_batch, reward_batch, done_batch, next_state_batch, optimizer)

    @tf.function
    def update_batch(self, state_batch, action_batch, reward_batch, done_batch, next_state_batch, optimizer):
        belief = tf.where(done_batch, reward_batch, reward_batch - tf.stop_gradient(tf.reduce_max(self.action_values_target(next_state_batch), axis=-1)))

        with tf.GradientTape() as tape:
            x = self.qnet(state_batch, training=True)
            ind = tf.concat(
                [tf.expand_dims(tf.range(0, action_batch.shape[0]), axis=-1), tf.expand_dims(action_batch, axis=-1)],
                axis=1)
            x = tf.gather_nd(x, ind)
            square = tf.square(x - belief)
            loss = tf.reduce_mean(square)
            tf.print(loss)

            grads = tape.gradient(loss, self.qnet.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.qnet.trainable_variables))

    def update_memory(self, sars):
        self.memory.push(sars)

    def memory_full(self):
        return self.memory.isfull()

    def clear_memory(self):
        self.memory.empty()

    def memory_size(self):
        return len(self.memory)

    def evaluate_from_memory(self, number_states,possible_actions_func, discount_factor=-1):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(number_states)
        state_batch = np.array(state_batch)

        belief = self.qnet.predict_on_batch(state_batch).numpy()
        for j in range(number_states):
            belief[j, action_batch[j]] = reward_batch[j] if done_batch[j] else reward_batch[j] + discount_factor * self.top_action(next_state_batch[j], possible_actions_func)[1]

        '''if loss is None:
        loss = self.qnet.evaluate(x=state_batch, y=belief)'''

        return self.qnet.evaluate(state_batch, belief, verbose=0)

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.qnet.save(dir_path + '/qnet.h5')
        self.target_net.save(dir_path + '/target_net.h5')
        self.memory.save(dir_path + '/memory.pickle')

    @staticmethod
    def load(file_path):
        net = load_model(file_path + '/qnet.h5')
        target_net = load_model(file_path + '/target_net.h5')
        memory = ConnectMemory.ValueFunctionMemory2.load(file_path + '/memory.pickle')
        return StateToActionAgent(qnet=net, target_net=target_net, memory=memory)


class ConnectAgentGPU(object):

    def __init__(self, live_net : tf.keras.Model, target_net: tf.keras.Model, memory: ConnectMemory.ValueFunctionMemory2):
        self.live_net = live_net
        self.target_net = target_net
        self.memory = memory

    def choose_random_action(self, state, possible_actions_func):
        possible_actions = possible_actions_func(state)
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def choose_greedy_action(self, state, possible_actions_func):
        return self.top_action_live(state, possible_actions_func)[0]

    def choose_eps_greedy_action(self, state, possible_action_func, eps=0.3):
        if random.random() < eps:
            return self.choose_random_action(state, possible_action_func)
        else:
            return self.choose_greedy_action(state, possible_action_func)

    @tf.function
    def action_value_live(self, state, action):
        action = tf.one_hot(tf.expand_dims(action, 0), depth=7)
        state = tf.expand_dims(state, 0)
        return self.live_net([state, action])

    @tf.function
    def top_action_value_target(self, state, possible_actions):
        state = tf.expand_dims(state, 0)
        one_hot_action = tf.one_hot(possible_actions, depth=7)
        num_options = possible_actions.shape[0]
        states = tf.tile(state, [num_options, 1, 1, 1])
        all_actions = self.target_net([states, one_hot_action])
        return tf.reduce_max(all_actions)

    def top_action_live(self, state, possible_actions_func):
        top_action = None
        top_action_value = None
        for action in possible_actions_func(state):
            action_value = self.action_value_live(tf.convert_to_tensor(state), tf.convert_to_tensor(action))
            if top_action is None:
                top_action = action
                top_action_value = action_value
            elif action_value > top_action_value:
                top_action = action
                top_action_value = action_value

        return top_action, top_action_value

    def top_action_values_target(self, states, possible_actions_func):
        next_state_values = []
        for state in states:
            possible_actions = possible_actions_func(state)
            next_state_value = self.top_action_value_target(tf.convert_to_tensor(state, dtype=tf.float32), tf.convert_to_tensor(possible_actions))
            next_state_values.append(next_state_value)
        return next_state_values

    def construct_targets(self, rewards, dones, next_states, possible_actions_func):

        next_state_values = self.top_action_values_target(next_states, possible_actions_func)
        return self.tf_construct_targets(tf.convert_to_tensor(rewards, dtype=tf.float32), tf.convert_to_tensor(dones, dtype=tf.bool), tf.convert_to_tensor(next_state_values, dtype=tf.float32))

    @tf.function
    def tf_construct_targets(self, rewards, dones, next_state_values):
        return tf.where(dones, rewards, rewards - next_state_values)

    def train(self, batch_size, optimizer, possible_actions_func):
        if len(self.memory) < batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)
        targets = self.construct_targets(reward_batch, done_batch, next_state_batch, possible_actions_func)
        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        targets = tf.stop_gradient(targets)
        self.apply_gradients(state_batch, targets, action_batch, optimizer)

    @tf.function
    def apply_gradients(self, states, targets, actions, optimizer):
        actions = tf.one_hot(actions, depth=7)
        with tf.GradientTape() as tape:
            x = self.live_net([states, actions], training=True)
            square = tf.square(x - tf.expand_dims(targets, axis=-1))
            loss = tf.reduce_mean(square)
            tf.print(loss)
        grads = tape.gradient(loss, self.live_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.live_net.trainable_variables))

    def memory_full(self):
        return self.memory.isfull()

    def update_memory(self, sars):
        self.memory.push(sars)

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.live_net.save(dir_path + '/live_net.h5')
        self.target_net.save(dir_path + '/target_net.h5')
        self.memory.save(dir_path + '/memory.pickle')

    @staticmethod
    def load(file_path):
        net = load_model(file_path + '/live_net.h5')
        target_net = load_model(file_path + '/target_net.h5')
        memory = ConnectMemory.ValueFunctionMemory2.load(file_path + '/memory.pickle')
        return ConnectAgentGPU(net, target_net, memory)



