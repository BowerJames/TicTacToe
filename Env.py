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

    def record_game_going_first(self, print_game=False):
        self.reset()
        self.Player_Turn = 1
        states = np.empty((0, 9))
        rewards = np.empty((0,))
        actions = np.empty((0,))

        while not self.Termination:
            states = np.append(states, np.expand_dims(self.State, 0), axis=0)

            action = self.Agent_1.choose_action(self.State, self.possible_actions())
            self.make_move(action, 1)
            actions = np.append(actions,action)
            if print_game:
                self.render()
                print('')
                print('---------------')
                print('')
            reward = self.evaluate_board()

            if not self.Termination:
                action = self.Agent_2.choose_action(-1*self.State, self.possible_actions())
                self.make_move(action, -1)
                if print_game:
                    self.render()
                    print('')
                    print('---------------')
                    print('')
                reward += self.evaluate_board()
            rewards = np.append(rewards, reward)

        return states, rewards, actions

    def discount_reward(self, rewards, gamma=0.95):
        dr = np.empty_like(rewards)
        reversed_rewards = np.flip(rewards)
        for i in range(len(reversed_rewards)):
            if i == 0:
                dr[i] = reversed_rewards[i]
            else:
                dr[i] = reversed_rewards[i] + gamma * dr[i-1]
        return np.flip(dr)

    def train_going_first(self, num_updates=50, episodes_per_batch=20):
        for batch in range(num_updates):
            if batch == 200:
                print(batch)
            batch_states = np.empty((0, 9))
            batch_rewards = np.empty((0,))
            batch_actions = np.empty((0,))
            for episode in range(episodes_per_batch):
                ep_states, ep_rewards, ep_actions = self.record_game_going_first()
                discounted_ep_rewards = self.discount_reward(ep_rewards)
                batch_states = np.append(batch_states, ep_states, axis=0)
                batch_rewards = np.append(batch_rewards, discounted_ep_rewards)
                batch_actions = np.append(batch_actions, ep_actions)
            loss = self.Agent_1.update_actor(batch_states, batch_actions, batch_rewards)

    def train(self, num_updates=50, episodes_per_batch=20):
        for batch in range(num_updates):
            if batch == 200:
                print(batch)
            batch_states = np.empty((0, 9))
            batch_rewards = np.empty((0,))
            batch_actions = np.empty((0,))
            for episode in range(episodes_per_batch):
                ep_states, ep_rewards, ep_actions = self.record_game()
                discounted_ep_rewards = self.discount_reward(ep_rewards)
                batch_states = np.append(batch_states, ep_states, axis=0)
                batch_rewards = np.append(batch_rewards, discounted_ep_rewards)
                batch_actions = np.append(batch_actions, ep_actions)
            self.Agent_1 = self.Agent_1.update_agent_2(batch_states, batch_actions, batch_rewards, self.possible_actions_from_state)

    def record_game(self, print_game=False):
        self.reset()
        self.Player_Turn = random.choice([-1, 1])
        states = np.empty((0, 9))
        rewards = np.empty((0,))
        actions = np.empty((0,))

        while not self.Termination:
            reward = 0
            if self.Player_Turn == 1:
                states = np.append(states, np.expand_dims(self.State, 0), axis=0)
                action = self.Agent_1.choose_action(self.State, self.possible_actions())
                self.make_move(action, 1)
                actions = np.append(actions, action)
                if print_game:
                    self.render()
                    print('')
                    print('---------------')
                    print('')
                reward += self.evaluate_board()

                self.Player_Turn *= -1

            if self.Player_Turn == -1:
                if not self.Termination:
                    action = self.Agent_2.choose_action(-1 * self.State, self.possible_actions())
                    self.make_move(action, -1)
                    if print_game:
                        self.render()
                        print('')
                        print('---------------')
                        print('')
                    reward += self.evaluate_board()
                self.Player_Turn *= -1

            if len(actions) > 0:
                rewards = np.append(rewards, reward)

        return states, rewards, actions

    def update_agent_2(self, new_player_2):
        self.Agent_2 = new_player_2

    def get_agent_1(self):
        return self.Agent_1

    def play_best_game(self):
        self.reset()
        self.Player_Turn = random.choice([-1, 1])

        while not self.Termination:
            if self.Player_Turn == 1:

                action = self.Agent_1.choose_top_action(self.State, self.possible_actions())
                self.make_move(action, 1)

                self.render()
                print('')
                print('---------------')
                print('')
                _ = self.evaluate_board()
                self.Player_Turn *= -1

            if self.Player_Turn == -1:
                if not self.Termination:
                    action = self.Agent_2.choose_top_action(-1 * self.State, self.possible_actions())
                    self.make_move(action, -1)
                    self.render()
                    print('')
                    print('---------------')
                    print('')
                    _ = self.evaluate_board()
                self.Player_Turn *= -1

    def calc_tot_rewards(self,number_games=100):
        tot_rewards = 0
        for i in range(number_games):
            _, rewards, _ = self.record_game()
            tot_rewards += rewards[-1]
        return tot_rewards