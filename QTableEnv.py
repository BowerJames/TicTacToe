import random
import numpy as np
from Agent import QTableAgent
from TicTacToeBoard import TicTacToeBoard


class QTableEnv(object):

    def __init__(self, agent1: QTableAgent, agent2: QTableAgent):
        self.agent1 = agent1
        self.agent2 = agent2

    def stats(self, num_games):
        player_1_wins = 0
        player_2_wins = 0
        draws = 0

        for i in range(num_games):
            sars1, _ = self.record_game()
            if sars1[-1][2] == 1:
                player_1_wins +=1
            elif sars1[-1][2] == -1:
                player_2_wins +=1
            else:
                draws +=1

        return player_1_wins, player_2_wins, draws



    def record_random_games(self, num_games):
        sars1 = []
        sars2 = []
        for i in range(num_games):
            epsars1, epsars2 = self.record_random_game()
            sars1 = sars1 + epsars1
            sars2 = sars2 + epsars2

        return sars1, sars2

    def record_games(self, num_games):
        sars1 = []
        sars2 = []
        for i in range(num_games):
            epsars1, epsars2 = self.record_game()
            sars1 = sars1 + epsars1
            sars2 = sars2 + epsars2

        return sars1, sars2

    def record_game(self):
        board = TicTacToeBoard()
        players_turn = random.choice([-1, 1])
        sars1 = []
        sars2 = []
        reward1 = None
        reward2 = None
        while not board.Termination:
            if players_turn == 1:
                state1 = board.get_state().astype(int)
                if len(sars1) > 0:
                    sars1[-1][2] = reward1
                    sars1[-1][3] = state1
                action1 = self.agent1.choose_action(state1, board.possible_actions())

                board.make_move(action1, 1)
                reward1 = board.evaluate_board()
                if reward2 is not None:
                    reward2 -= reward1
                sars1.append([state1, action1, None, None])
                if board.Termination:
                    sars1[-1][2] = reward1
                    sars2[-1][2] = reward2
                players_turn *= -1

            elif players_turn == -1:
                state2 = board.get_state().astype(int)
                if len(sars2) > 0:
                    sars2[-1][2] = reward2
                    sars2[-1][3] = state2
                action2 = self.agent2.choose_action(state2, board.possible_actions())
                board.make_move(action2, -1)
                reward2 = -1 * board.evaluate_board()
                if reward1 is not None:
                    reward1 -= reward2
                sars2.append([state2, action2, None, None])
                if board.Termination:
                    sars1[-1][2] = reward1
                    sars2[-1][2] = reward2
                players_turn *= -1

        return sars1, sars2


    def record_random_game(self):
        board = TicTacToeBoard()
        players_turn = random.choice([-1, 1])
        sars1 = []
        sars2 = []
        reward1 = None
        reward2 = None
        while not board.Termination:
            if players_turn == 1:
                state1 = board.get_state().astype(int)
                if len(sars1) > 0:
                    sars1[-1][2] = reward1
                    sars1[-1][3] = state1
                action1 = self.agent1.choose_random_action(board.possible_actions())

                board.make_move(action1, 1)
                reward1 = board.evaluate_board()
                if reward2 is not None:
                    reward2 -= reward1
                sars1.append([state1, action1, None, None])
                if board.Termination:
                    sars1[-1][2] = reward1
                    sars2[-1][2] = reward2
                players_turn *= -1

            elif players_turn == -1:
                state2 = board.get_state().astype(int)
                if len(sars2) > 0:
                    sars2[-1][2] = reward2
                    sars2[-1][3] = state2
                action2 = self.agent2.choose_random_action(board.possible_actions())
                board.make_move(action2, -1)
                reward2 = -1 * board.evaluate_board()
                if reward1 is not None:
                    reward1 -= reward2
                sars2.append([state2, action2, None, None])
                if board.Termination:
                    sars1[-1][2] = reward1
                    sars2[-1][2] = reward2
                players_turn *= -1

        return sars1, sars2

    def train(self, num_iter, num_episodes):
        for i in range(num_iter):
            sars1, _ = self.record_games(num_episodes)
            self.agent1.train(sars1)


    def print_game(self):
        board = TicTacToeBoard()
        players_turn = random.choice([-1, 1])
        print("Player {0} to start\n".format(players_turn, ))
        board.render()
        print("------------------")

        while not board.Termination:
            if players_turn == 1:
                action = self.agent1.choose_action(board.get_state(), board.possible_actions())
                board.make_move(action, players_turn)
                players_turn *= -1
                board.render()
                print("------------------")
                _ = board.evaluate_board()
            elif players_turn == -1:
                action = self.agent2.choose_action(board.get_state(), board.possible_actions())
                board.make_move(action, players_turn)
                players_turn *= -1
                board.render()
                print("------------------")
                _ = board.evaluate_board()





