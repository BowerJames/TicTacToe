import random
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from Networks import ValueNetwork, PolicyNetwork
from tensorflow.keras.optimizers import Optimizer
import pickle
from TicTacMemory import ValueFunctionMemory
import os
from tensorflow.keras.models import load_model
import Policy_Memory
import Networks
import tensorflow.keras.optimizers as Optimizers


class Agent(ABC):

    @abstractmethod
    def choose_action(self, state, possible_actions):
        pass


class RandomAgent(Agent):

    def choose_action(self, state, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options-1)]

    def choose_top_action(self, state, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def choose_random_action(self, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]


class ValueAgent(Agent):
    def __init__(self, value_net: ValueNetwork, possible_action_function, optimizer: Optimizer, memory: ValueFunctionMemory=None, eps=0.3, discount_factor=0.95):
        self.value_network = value_net
        self.eps = eps
        self.action_function = possible_action_function
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        if memory is None:
            self.memory = ValueFunctionMemory()
        else:
            self.memory = memory

    def compile_net(self):
        self.value_network.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def choose_random_action(self, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def choose_action(self, state, possible_actions):
        rnd = random.random()
        if rnd < self.eps:
            return self.choose_random_action(possible_actions)

        return self.top_action(state)[0]

    def add_to_memory(self, sars):
        self.memory.push(sars)

    def train(self, num_batches, batch_size):
        for i in range(num_batches):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)
            state_batch = np.asarray(state_batch)
            action_batch = np.asarray(action_batch)
            reward_batch = np.asarray(reward_batch)
            next_state_batch = np.asarray(next_state_batch)
            done_batch = np.asarray(done_batch)

            targets = reward_batch + (1-done_batch) * self.discount_factor * self.top_action_values(next_state_batch)
            state_actions = np.append(state_batch, action_batch, axis=1)
            self.value_network.fit(state_actions, targets, batch_size=batch_size)


    def top_action_values(self, states):
        top_action_values = np.empty((0,))
        for state in states:
            if state is None:
                top_action_values = np.append(top_action_values, 0)
            else:
                top_action_values = np.append(top_action_values, self.top_action(np.expand_dims(state, axis=0))[1])
        return top_action_values


    def top_action(self, state):
        top_action = None
        top_action_value = None
        for action in self.action_function(state):
            one_hot_action = np.zeros((1, 9))
            one_hot_action[0, action] = 1
            state_action = np.append(state, one_hot_action, axis=1)
            action_value = self.value_network.predict_on_batch(state_action)
            if top_action is None:
                top_action = action
                top_action_value = action_value
            elif action_value > top_action_value:
                top_action = action
                top_action_value = action_value

        return top_action, top_action_value

    def save(self, weight_name, memory_name):
        self.value_network.save_weights(weight_name)
        file = open(memory_name + '.pickle', 'wb')
        pickle.dump(self.memory, file)

    @staticmethod
    def load(weight_name, memory_name, inp_size, hidden, possible_action_function, optimizer, eps=0.3, reg_coef=0.01):
        net = ValueNetwork(inp_size, hidden, reg_coef=reg_coef)
        net.load_weights(weight_name)

        file = open(memory_name + '.pickle', 'rb')
        memory = pickle.load(file)

        return ValueAgent(net, possible_action_function, optimizer, memory=memory, eps=eps)

class PolicyAgent(Agent):

    def __init__(self, policy_net: PolicyNetwork, optimizer: Optimizer, possible_actions_func ):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.possible_actions_func = possible_actions_func

    def choose_action(self, state, possible_actions):
        fake_logits = self.policy_net.predict_on_batch(state)
        fake_logits_reshaped = tf.squeeze(fake_logits)
        real_logits = tf.gather(fake_logits_reshaped, possible_actions)
        prob = tf.nn.softmax(real_logits)
        action = tf.random.categorical(tf.math.log(tf.expand_dims(prob, axis=0)), 1)
        return possible_actions[action]

    def train_on_batch(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            fake_logits = self.policy_net.predict_on_batch(states)
            fake_logits_reshaped = tf.squeeze(fake_logits)

class ActorCriticAgent(Agent):

    def __init__(self, policy_network: PolicyNetwork, value_network: ValueNetwork, policy_opt: Optimizer, value_opt: Optimizer, eps):
        self.policy_network = policy_network
        self.value_network = value_network
        self.policy_opt = policy_opt
        self.value_opt = value_opt
        self.policy_data = np.empty((0, 18))
        self.eps = eps
        self.value_train_params = {'Epochs': 10, 'Batch_Size': 30}
        self.policy_train_params = {'Epochs': 10, 'Batch_Size': 30}


    def choose_action(self, state, possible_actions):
        pass

    def set_value_train_params(self, epochs, batch_size):
        self.value_train_params['Epochs'] = epochs
        self.value_train_params['Batch_Size'] = batch_size

    def set_policy_train_params(self, epochs, batch_size):
        self.policy_train_params['Epochs'] = epochs
        self.policy_train_params['Batch_Size'] = batch_size

    def compile_policy_network(self):
        self.policy_network.compile(optimizer=self.policy_opt, loss=tf.keras.losses.CategoricalCrossentropy())

    def compile_value_network(self):
        self.value_network.compile(optimizer=self.value_opt, loss=tf.keras.losses.MeanSquaredError())

    def choose_policy_action(self, state, possible_actions):
        x = self.policy_network.predict_on_batch(state)
        x = tf.squeeze(x)
        x = tf.gather(x, possible_actions)
        x = tf.nn.softmax(x)
        x = tf.random.categorical(tf.math.log(tf.expand_dims(x, axis=0)), 1)
        return possible_actions[x]

    def choose_random_action(self, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def choose_eps_greedy(self, state, possible_actions):
        top_action = None
        top_value = None
        rnd = random.random()
        if rnd < self.eps:
            return self.choose_random_action(possible_actions)

        for action in possible_actions:
            one_hot_action = np.zeros((1, 9))
            one_hot_action[:, action] = 1
            inp = np.append(state, one_hot_action, axis=1)
            prediction = self.value_network.predict_on_batch(inp)
            if top_action is None:
                top_action = action
                top_value = prediction
            elif prediction > top_value:
                top_action = action
                top_value = prediction

        return top_action

    def update_value_net(self, states, actions, rewards):
        int_actions = actions.astype(int).reshape(-1)
        one_hot_actions = np.eye(9)[int_actions]
        x_train = np.append(states, one_hot_actions, axis=1)
        self.value_network.fit(x_train, rewards, batch_size=self.value_train_params['Batch_Size'], epochs=self.value_train_params['Epochs'])
        return self.value_network.evaluate(x_train, rewards, batch_size=self.value_train_params['Batch_Size'])

    def add_to_policy_memory(self, states, actions):
        int_actions = actions.astype(int).reshape(-1)
        one_hot_actions = np.eye(9)[int_actions]
        new_data = np.append(states, one_hot_actions, axis=1)
        self.policy_data = np.append(self.policy_data, new_data, axis=0)
        np.random.shuffle(self.policy_data)

    def train_policy_net(self):
        states = self.policy_data[:, :9]
        actions = self.policy_data[:, 9:]
        self.policy_network.fit(states, actions, batch_size=self.policy_train_params['Batch_Size'], epochs=self.policy_train_params['Epochs'])

    def save(self, name):
        self.policy_network.save_weights(name + 'Policy.h5')
        self.value_network.save_weights(name + 'Value.h5')


class QTableAgent(Agent):

    def __init__(self, poss_actions_func):
        self.state_dict = {}
        self.update_count = 1
        self.poss_actions_func = poss_actions_func

    def choose_random_action(self, possible_actions):
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def serialize_state_action(self, state, action):
        string = ''
        for digit in state[0].astype(int):
            string = string + str(digit)
        string = string + str(action)
        return string

    def train(self, sars):
        for state, action, reward, next_state in sars:
            state_action_serial = self.serialize_state_action(state, action)
            if state_action_serial not in self.state_dict.keys():
                self.state_dict[state_action_serial] = 0
            if next_state is None:
                self.state_dict[state_action_serial] = (1 - 1 / self.update_count) * self.state_dict[state_action_serial] + (1 / self.update_count) * reward
            else:
                next_poss_actions = self.poss_actions_func(next_state)
                top_action = None
                top_action_value = None
                for next_action in next_poss_actions:
                    next_state_action_serial = self.serialize_state_action(next_state, next_action)
                    if next_state_action_serial not in self.state_dict.keys():
                        self.state_dict[next_state_action_serial] = 0
                    next_action_value = self.state_dict[next_state_action_serial]
                    if top_action is None:
                        top_action = next_state_action_serial
                        top_action_value = next_action_value
                    elif next_action_value > top_action_value:
                        top_action = next_state_action_serial
                        top_action_value = next_action_value
                self.state_dict[state_action_serial] = (1 - 1 / self.update_count) * self.state_dict[state_action_serial] + (1 / self.update_count) * (reward + top_action_value)

    def increase_update_count(self):
        self.update_count += 1

    def choose_action(self, state, possible_actions):
        top_action = None
        top_action_value = None
        for action in possible_actions:
            state_action_serial = self.serialize_state_action(state, action)
            if state_action_serial not in self.state_dict.keys():
                self.state_dict[state_action_serial] = 0
            next_action_value = self.state_dict[state_action_serial]
            if top_action is None:
                top_action = action
                top_action_value = next_action_value
            elif next_action_value > top_action_value:
                top_action = action
                top_action_value = next_action_value

        return top_action

    def save(self, filename):
        file = open(filename + '.pickle', 'wb')
        pickle.dump(self, file)

    @staticmethod
    def load(filename):
        file = open(filename + '.pickle', 'rb')
        return pickle.load(file)


class EnsembleValueAgent(object):

    def __init__(self, live_agent, agent_memory):

        self.live_agent = live_agent
        self.agent_memory = agent_memory

    def choose_random_action(self, state, possible_actions_func):
        possible_actions = possible_actions_func(state)
        num_options = len(possible_actions)
        return possible_actions[random.randint(0, num_options - 1)]

    def choose_policy_action(self, state, possible_actions_func):
        if len(self.agent_memory) == 0:
            return self.choose_random_action(state, possible_actions_func)
        else:
            chosen_agent = self.agent_memory.sample()
            return chosen_agent.choose_greedy_action(state, possible_actions_func)

    def choose_live_eps_greedy_action(self, state, possible_actions_func, eps=0.3):
        return self.live_agent.choose_eps_greedy_action(state, possible_actions_func, eps=eps)

    def update_live_memory(self, sars):
        self.live_agent.add_to_memory(sars)

    def train_live_agent(self, num_batches, batch_size, possible_actions_func, epochs=1, discount_factor=0.99):
        self.live_agent.train(num_batches, batch_size, possible_actions_func, epochs=epochs, discount_factor=discount_factor)

    def train_live_agent_with_knowledge(self, num_batches, batch_size, possible_actions_func, epochs=1, discount_factor=-1):
        self.live_agent.train_with_knowledge(num_batches, batch_size, possible_actions_func, epochs=epochs, discount_factor=discount_factor)

    def update_policy(self, inp_dim, h_layers, reg_coef=0.01, learning_rate=0.01, new_agent_max_memory=50000):
        self.agent_memory.push(self.live_agent)
        new_net = Networks.value_network(inp_dim, h_layers, reg_coef=reg_coef)
        new_net.compile(optimizer=Optimizers.SGD(learning_rate=learning_rate), loss='mean_squared_error')
        new_agent_memory = ValueFunctionMemory(max_memory=new_agent_max_memory)
        new_agent = IndividualValueAgent(qnet=new_net, memory=new_agent_memory)
        self.live_agent = new_agent

    def save(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.live_agent.save(dir + '/live_agent')
        self.agent_memory.save(dir + '/agent_memory')

    @staticmethod
    def load( dir):
        live_agent = IndividualValueAgent.load(dir + '/live_agent')
        agent_memory = Policy_Memory.PolicyMemory.load(dir + '/agent_memory')
        return EnsembleValueAgent(live_agent, agent_memory)


class IndividualValueAgent(object):

    def __init__(self, qnet=None, memory=None):
        self.qnet = qnet
        self.memory = memory

    def memory_full(self):
        return self.memory.isfull()

    def refresh_memory(self):
        self.memory.empty()

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

    def top_action(self, state, possible_actions_func):
        top_action = None
        top_action_value = None
        for action in possible_actions_func(state):
            one_hot_action = np.zeros((1, 9))
            one_hot_action[0, action] = 1
            state_action = np.append(state, one_hot_action, axis=1)
            action_value = self.qnet.predict_on_batch(state_action)
            if top_action is None:
                top_action = action
                top_action_value = action_value
            elif action_value > top_action_value:
                top_action = action
                top_action_value = action_value

        return top_action, top_action_value

    def bottom_action(self, state, possible_actions_func):
        bottom_acton = None
        bottom_acton_value = None

        for action in possible_actions_func(state):
            one_hot_action = np.zeros((1, 9))
            one_hot_action[0, action] = 1
            state_action = np.append(state, one_hot_action, axis=1)
            action_value = self.qnet.predict_on_batch(state_action)
            if bottom_acton is None:
                bottom_acton = action
                bottom_acton_value = action_value
            elif action_value < bottom_acton_value:
                bottom_acton = action
                bottom_acton_value = action_value
        return bottom_acton, bottom_acton_value

    def add_to_memory(self, sars):
        self.memory.push(sars)

    def train(self, number_states, batch_size, possible_actions_func, epochs=1, discount_factor=0.99):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(number_states)
        state_batch = np.asarray(state_batch)
        action_batch = np.asarray(action_batch)
        reward_batch = np.asarray(reward_batch)
        next_state_batch = np.asarray(next_state_batch)
        done_batch = np.asarray(done_batch)

        state_actions = np.append(state_batch, action_batch, axis=1)
        #targets = np.squeeze(self.qnet.predict_on_batch(state_actions).numpy()) * (1-learning_rate) + learning_rate * (reward_batch + (1-done_batch) * discount_factor * self.top_action_values(next_state_batch, possible_actions_func))
        targets = reward_batch + (1-done_batch) * discount_factor * self.top_action_values(next_state_batch, possible_actions_func)
        self.qnet.compile(optimizer=Optimizers.SGD(learning_rate=0.2), loss='mean_squared_error')
        self.qnet.fit(state_actions, targets, batch_size=batch_size, epochs=epochs)
        self.qnet.compile(optimizer=Optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')
        self.qnet.fit(state_actions, targets, batch_size=batch_size, epochs=epochs)

    def train_with_knowledge(self, number_states, batch_size, possible_actions_func, epochs=1, discount_factor=-1):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(number_states)
        state_batch = np.asarray(state_batch)
        action_batch = np.asarray(action_batch)
        reward_batch = np.asarray(reward_batch)
        next_state_batch = np.asarray(next_state_batch)
        done_batch = np.asarray(done_batch)

        state_actions = np.append(state_batch, action_batch, axis=1)
        #targets = np.squeeze(self.qnet.predict_on_batch(state_actions).numpy()) * (1-learning_rate) + learning_rate * (reward_batch + (1-done_batch) * discount_factor * self.top_action_values(next_state_batch, possible_actions_func))
        targets = reward_batch + (1-done_batch) * discount_factor * self.top_action_values(next_state_batch, possible_actions_func)
        self.qnet.compile(optimizer=Optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')
        self.qnet.fit(state_actions, targets, batch_size=batch_size, epochs=epochs)
        self.qnet.compile(optimizer=Optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')
        self.qnet.fit(state_actions, targets, batch_size=batch_size, epochs=epochs)

    def top_action_values(self, states, possible_actions_func):
        top_action_values = np.empty((0,))
        for state in states:
            if state is None:
                top_action_values = np.append(top_action_values, 0)
            else:
                top_action_values = np.append(top_action_values, self.top_action(np.expand_dims(state, axis=0), possible_actions_func)[1])
        return top_action_values

    def bottom_action_values(self, states, possible_actions_func):
        bottom_action_values = np.empty((0,))
        for state in states:
            if state is None:
                bottom_action_values = np.append(bottom_action_values, 0)
            else:
                bottom_action_values = np.append(bottom_action_values, self.bottom_action(np.expand_dims(state, axis=0), possible_actions_func)[1])
        return bottom_action_values

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.qnet.save(dir_path + '/qnet.h5')
        self.memory.save(dir_path + '/memory.pickle')

    @staticmethod
    def load(file_path):
        net = load_model(file_path + '/qnet.h5')
        memory = ValueFunctionMemory.load(file_path + '/memory.pickle')
        return IndividualValueAgent(qnet=net, memory=memory)

class TemporalDifferenceAgent(object):

    def __init__(self, net, possible_action_func):
        self.net = net
        self.state_dict = {}
        self.update_count = 1
        self.possible_action_func = possible_action_func

























