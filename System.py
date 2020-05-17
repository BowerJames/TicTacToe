import copy
import random
import TicTacToeBoard
import Agent
import numpy as np


class System(object):

    def __init__(self, agent_1: Agent.Agent, agent_2: Agent.Agent):
        self.agent_1 = agent_1
        self.agent_2 = agent_2

    def print_game(self):
        board = TicTacToeBoard.TicTacToeBoard()
        players_turn = random.choice([-1, 1])
        print("Player {0} to start\n".format(players_turn,))
        board.render()
        print("------------------")

        while not board.Termination:
            if players_turn == 1:
                action = self.agent_1.choose_action(board.get_state(), board.possible_actions())
                board.make_move(action, players_turn)
                players_turn *= -1
                board.render()
                print("------------------")
                _ = board.evaluate_board()
            elif players_turn == -1:
                action = self.agent_2.choose_random_action(-1*board.get_state(), board.possible_actions())
                board.make_move(action, players_turn)
                players_turn *= -1
                board.render()
                print("------------------")
                _ = board.evaluate_board()

    def record_game(self):
        board = TicTacToeBoard.TicTacToeBoard()
        players_turn = random.choice([-1, 1])
        player_1_states = np.empty((0, 9))
        player_2_states = np.empty((0, 9))
        player_1_actions = np.empty((0,))
        player_2_actions = np.empty((0,))
        player_1_rewards = np.empty((0,))
        player_2_rewards = np.empty((0,))
        player_1_reward = None
        player_2_reward = None
        while not board.Termination:
            if players_turn == 1:
                player_1_states = np.append(player_1_states, board.get_state(), axis=0)
                action = self.agent_1.choose_action(board.get_state(), board.possible_actions())
                player_1_actions = np.append(player_1_actions, action)
                board.make_move(action, 1)
                player_1_reward = board.evaluate_board()
                if player_2_reward is not None:
                    player_2_reward -= player_1_reward
                    player_2_rewards = np.append(player_2_rewards, player_2_reward)
                if board.Termination:
                    player_1_rewards = np.append(player_1_rewards, player_1_reward)
                players_turn *= -1

            elif players_turn == -1:
                player_2_states = np.append(player_2_states, -1 * board.get_state(), axis=0)
                action = self.agent_2.choose_action(-1 * board.get_state(), board.possible_actions())
                player_2_actions = np.append(player_2_actions, action)
                board.make_move(action, -1)
                player_2_reward = -1 * board.evaluate_board()
                if player_1_reward is not None:
                    player_1_reward -= player_2_reward
                    player_1_rewards = np.append(player_1_rewards, player_1_reward)
                if board.Termination:
                    player_2_rewards = np.append(player_2_rewards, player_2_reward)
                players_turn *= -1

        return player_1_states, player_1_actions, player_1_rewards, player_2_states, player_2_actions, player_2_rewards

    def record_top_game(self):
        board = TicTacToeBoard.TicTacToeBoard()
        players_turn = random.choice([-1, 1])
        player_1_states = np.empty((0, 9))
        player_2_states = np.empty((0, 9))
        player_1_actions = np.empty((0,))
        player_2_actions = np.empty((0,))
        player_1_rewards = np.empty((0,))
        player_2_rewards = np.empty((0,))
        player_1_reward = None
        player_2_reward = None
        while not board.Termination:
            if players_turn == 1:
                player_1_states = np.append(player_1_states, board.get_state(), axis=0)
                action = self.agent_1.choose_top_action(board.get_state(), board.possible_actions())
                player_1_actions = np.append(player_1_actions, action)
                board.make_move(action, 1)
                player_1_reward = board.evaluate_board()
                if player_2_reward is not None:
                    player_2_reward -= player_1_reward
                    player_2_rewards = np.append(player_2_rewards, player_2_reward)
                if board.Termination:
                    player_1_rewards = np.append(player_1_rewards, player_1_reward)
                players_turn *= -1

            elif players_turn == -1:
                player_2_states = np.append(player_2_states, -1 * board.get_state(), axis=0)
                action = self.agent_2.choose_top_action(-1 * board.get_state(), board.possible_actions())
                player_2_actions = np.append(player_2_actions, action)
                board.make_move(action, -1)
                player_2_reward = -1 * board.evaluate_board()
                if player_1_reward is not None:
                    player_1_reward -= player_2_reward
                    player_1_rewards = np.append(player_1_rewards, player_1_reward)
                if board.Termination:
                    player_2_rewards = np.append(player_2_rewards, player_2_reward)
                players_turn *= -1

        return player_1_states, player_1_actions, player_1_rewards, player_2_states, player_2_actions, player_2_rewards

    def record_top_games(self, num_games):
        player_1_states = np.empty((0, 9))
        player_1_actions = np.empty((0, 9))
        player_1_rewards = np.empty((0,))
        player_2_states = np.empty((0, 9))
        player_2_actions = np.empty((0, 9))
        player_2_rewards = np.empty((0,))

        for game in range(num_games):
            ep_player_1_states, ep_player_1_actions, ep_player_1_rewards, ep_player_2_states, ep_player_2_actions, ep_player_2_rewards = self.record_top_game()
            ep_player_1_discount_rewards = self.discount_reward(ep_player_1_rewards)
            ep_player_2_discount_rewards = self.discount_reward(ep_player_2_rewards)

            player_1_states = np.append(player_1_states, ep_player_1_states, axis=0)
            player_1_actions = np.append(player_1_actions, ep_player_1_actions)
            player_1_rewards = np.append(player_1_rewards, ep_player_1_discount_rewards)
            player_2_states = np.append(player_2_states, ep_player_2_states, axis=0)
            player_2_actions = np.append(player_2_actions, ep_player_2_actions)
            player_2_rewards = np.append(player_2_rewards, ep_player_2_discount_rewards)

        return player_1_states, player_1_actions, player_1_rewards, player_2_states, player_2_actions, player_2_rewards

    def record_games(self, num_games):

        player_1_states = np.empty((0, 9))
        player_1_actions = np.empty((0, 9))
        player_1_rewards = np.empty((0,))
        player_2_states = np.empty((0, 9))
        player_2_actions = np.empty((0, 9))
        player_2_rewards = np.empty((0,))

        for game in range(num_games):
            ep_player_1_states, ep_player_1_actions, ep_player_1_rewards, ep_player_2_states, ep_player_2_actions, ep_player_2_rewards = self.record_game()
            ep_player_1_discount_rewards = self.discount_reward(ep_player_1_rewards)
            ep_player_2_discount_rewards = self.discount_reward(ep_player_2_rewards)

            player_1_states = np.append(player_1_states, ep_player_1_states, axis=0)
            player_1_actions = np.append(player_1_actions, ep_player_1_actions)
            player_1_rewards = np.append(player_1_rewards, ep_player_1_discount_rewards)
            player_2_states = np.append(player_2_states, ep_player_2_states, axis=0)
            player_2_actions = np.append(player_2_actions, ep_player_2_actions)
            player_2_rewards = np.append(player_2_rewards, ep_player_2_discount_rewards)

        return player_1_states, player_1_actions, player_1_rewards, player_2_states, player_2_actions, player_2_rewards

    def train_agent_1(self, num_update_stages, games_per_update, epochs, batch_size, from_scratch=False, print_stats=False):
        if print_stats:
            player_1_wins = []
            player_2_wins = []
            draws = []
        for update in range(num_update_stages):
            player_1_states = np.empty((0, 9))
            player_1_actions = np.empty((0, 9))
            player_1_rewards = np.empty((0,))

            if update == 0 and from_scratch:
                newsys = System(Agent.RandomAgent(), Agent.RandomAgent())
                player_1_states, player_1_actions, player_1_rewards, _, _, _ = newsys.record_games(games_per_update)
            else:
                for game in range(games_per_update):
                    ep_player_1_states, ep_player_1_actions, ep_player_1_rewards, _, _, _ = self.record_game()
                    ep_player_1_discount_rewards = self.discount_reward(ep_player_1_rewards)
                    player_1_states = np.append(player_1_states, ep_player_1_states, axis=0)
                    player_1_actions = np.append(player_1_actions, ep_player_1_actions)
                    player_1_rewards = np.append(player_1_rewards, ep_player_1_discount_rewards)
            if print_stats:
                up_p1_wins, up_p2_wins, up_draws = self.get_stats()
                player_1_wins.append(up_p1_wins)
                player_2_wins.append(up_p2_wins)
                draws.append(up_draws)
            self.agent_1.train(player_1_states, player_1_actions, player_1_rewards, epochs, batch_size)

        if print_stats:
            for i in range(len(player_1_wins)):
                print('--------------------')
                print('Stats for Update {0}'.format(i))
                print('Player 1 Wins: {0}'.format(player_1_wins[i]))
                print('Player 2 Wins: {0}'.format(player_2_wins[i]))
                print('Draws: {0}'.format(draws[i]))

        return self.agent_1


    def discount_reward(self, rewards, gamma=1):
        dr = np.empty_like(rewards)
        reversed_rewards = np.flip(rewards)
        for i in range(len(reversed_rewards)):
            if i == 0:
                dr[i] = reversed_rewards[i]
            else:
                dr[i] = reversed_rewards[i] + gamma * dr[i-1]
        return np.flip(dr)

    def get_stats(self, number_games=100):
        player_1_wins = 0
        players_2_wins = 0
        draws = 0
        for i in range(number_games):
            _, _, ep_player_1_rewards, _, _, _ = self.record_game()
            if ep_player_1_rewards[-1] == 1:
                player_1_wins += 1
            elif ep_player_1_rewards[-1] == -1:
                players_2_wins += 1
            elif ep_player_1_rewards[-1] == 0:
                draws += 1

        return player_1_wins, players_2_wins, draws

    def copy_agent_1(self):
        return self.agent_1.copy()

    def replace_agent_2(self, agent: Agent.Agent):
        self.agent_2 = agent


