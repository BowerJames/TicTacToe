import random
from collections import deque
import os
import numpy as np
import pickle


class ValueFunctionMemory(object):

    def __init__(self, max_memory=50000):
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)

    def push(self, sars):
        for state, action, reward, next_state in sars:
            one_hot_action = np.zeros((6, 7, 1))
            one_hot_action[:, action, 0] = 1
            self.memory.append((state, one_hot_action, reward, next_state, True if next_state is None else False))

    def isfull(self):
        return len(self.memory) == self.max_memory

    def empty(self):
        self.memory = deque(maxlen=self.max_memory)

    def __len__(self):
        return len(self.memory)


    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            memory = pickle.load(file)
        return memory


class ValueFunctionMemory2(object):

    def __init__(self, max_memory=50000):
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)


    def push(self, sars):
        for state, action, reward, next_state in sars:
            self.memory.append((state, action, reward, np.zeros((6, 7, 3)) if next_state is None else next_state, True if next_state is None else False))

    def isfull(self):
        return len(self.memory) == self.max_memory

    def empty(self):
        self.memory = deque(maxlen=self.max_memory)

    def __len__(self):
        return len(self.memory)


    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            memory = pickle.load(file)
        return memory