import numpy as np


class TicTacToeBoard(object):

    def __init__(self):
        self.state = np.zeros((9,))
        self.symbol_dic = {-1: 'X', 0: ' ', 1: 'O'}
        self.termination = False

    def render(self):
        for i in [0,3,6]:
            print(self.symbol_dic[int(self.state[i])] + '|' + self.symbol_dic[int(self.state[i+1])] + '|' + self.symbol_dic[int(self.state[i+2])])

    def possible_actions(self):
        return np.where(self.state == 0)[0]

    def make_move(self, square, players_turn):
        self.state[square] = 1 * players_turn

    def evaluate_board(self):
        lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for line in lines:
            if np.sum(self.state[line]) == 3:
                return 1, True
            if np.sum(self.state[line]) == -3:
                return -1, True
        if len(self.possible_actions()) == 0:
            return 0, True
        return None, False
