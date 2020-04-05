import random
import numpy as np
import tensorflow as tf
import Agent
import TicTacToeBoard


class Environment(TicTacToeBoard.TicTacToeBoard):

    def __init__(self, agent_1, agent_2):
        super().__init__()
        self.Agent_1 = agent_1
        self.Agent_2 = agent_2
        self.Player_Turn = 1

    def play_game(self):
        while not self.Termination:
            if self.Player_Turn == 1:
                action = self.Agent_1.choose_action(self.State, self.possible_actions())
                self.make_move(action, 1)
                reward = self.evaluate_board()
            elif self.Player_Turn == -1:
                action = self.Agent_2.choose_action(self.State, self.possible_actions())
                self.make_move(action, -1)
                reward = self.evaluate_board()
            self.render()
            self.Player_Turn *= -1





