import random
from collections import deque
import Agent
import os


class PolicyMemory(object):

    def __init__(self, max_memory=500):
        self.max_memory = max_memory
        self.memory = deque(maxlen=max_memory)

    def __len__(self):
        return len(self.memory)

    def push(self, agent):
        self.memory.append(agent)

    def sample(self):
        return random.choice(self.memory)

    def save(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

        for i in range(len(self.memory)):
            self.memory[i].save(dir + '/agent' + '_' + str(i))

    @staticmethod
    def load(path):
        model_files = os.listdir(path + '/')
        p_mem = PolicyMemory()
        for file in model_files:
            p_mem.push(Agent.IndividualValueAgent.load(path + '/' + file))

        return p_mem