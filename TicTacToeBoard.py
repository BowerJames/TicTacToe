import numpy as np


class TicTacToeBoard(object):

    def __init__(self):
        self.State = np.zeros((9,))
        self.Symbol_Dic = {-1: 'X', 0: ' ', 1: 'O'}
        self.Termination = False

    def reset(self):
        self.State = np.zeros((9,))
        self.Termination = False

    def render(self):
        for i in [0, 3, 6]:
            print(self.Symbol_Dic[int(self.State[i])] + '|' + self.Symbol_Dic[int(self.State[i + 1])] + '|' + self.Symbol_Dic[int(self.State[i + 2])])

    def possible_actions(self):
        return np.where(self.State == 0)[0]

    def make_move(self, square, players_turn):
        self.State[square] = 1 * players_turn

    def evaluate_board(self):
        lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for line in lines:
            if np.sum(self.State[line]) == 3:
                self.Termination = True
                return 1
            if np.sum(self.State[line]) == -3:
                self.Termination = True
                return -1
        if len(self.possible_actions()) == 0:
            self.Termination = True
            return 0
        return 0

    @staticmethod
    def possible_actions_from_state(state):
        return np.where(state[0] == 0)[0]

    def get_state(self):
        return np.expand_dims(self.State, 0)
