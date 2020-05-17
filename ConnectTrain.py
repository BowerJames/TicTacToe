import random

import numpy as np
import Connect4
import tensorflow as tf


def record_eps_greedy_self_play_game(model, eps1=0.5, eps2=0.5, print_game=False):
    sars1 = []
    sars2 = []
    players_turn = 1

    state, _ = Connect4.start_state()
    while not Connect4.evaluate_state(state)[1]:
        if players_turn == 1:
            if len(sars2) > 0:
                sars2[-1][3] = state
            turn = [state, None, None, None]
            action = model.choose_eps_greedy_action(state, Connect4.possible_actions, eps=eps1)
            state = Connect4.take_action(state, action, 1)
            if print_game:
                Connect4.print_state(state)
                print('')
                print('----------------------')
                print('')
            reward, _ = Connect4.evaluate_state(state)
            turn[1] = action
            turn[2] = reward
            sars1.append(turn)

            players_turn = 2

        elif players_turn == 2:
            if len(sars1) > 0:
                sars1[-1][3] = Connect4.reverse_state(state)
            turn = [Connect4.reverse_state(state), None, None, None]
            action = model.choose_eps_greedy_action(Connect4.reverse_state(state), Connect4.possible_actions, eps=eps2)
            state = Connect4.take_action(state, action, 2)
            if print_game:
                Connect4.print_state(state)
                print('')
                print('----------------------')
                print('')
            reward = -1 * Connect4.evaluate_state(state)[0]
            turn[1] = action
            turn[2] = reward
            sars2.append(turn)

            players_turn = 1

    return sars1, sars2


def record_eps_greedy_random_training_game(model, eps):
    sars1 = []
    sars2 = []
    players_turn = 1

    state, _ = Connect4.start_state()
    while not Connect4.evaluate_state(state)[1]:
        if players_turn == 1:
            if len(sars2) > 0:
                sars2[-1][3] = state
            turn = [state, None, None, None]
            action = model.choose_eps_greedy_action(state, Connect4.possible_actions, eps=eps)
            state = Connect4.take_action(state, action, 1)

            reward, _ = Connect4.evaluate_state(state)
            turn[1] = action
            turn[2] = reward
            sars1.append(turn)

            players_turn = 2

        elif players_turn == 2:
            if len(sars1) > 0:
                sars1[-1][3] = Connect4.reverse_state(state)
            turn = [Connect4.reverse_state(state), None, None, None]
            action = model.choose_random_action(Connect4.reverse_state(state), Connect4.possible_actions)
            state = Connect4.take_action(state, action, 2)

            reward = -1 * Connect4.evaluate_state(state)[0]
            turn[1] = action
            turn[2] = reward
            sars2.append(turn)

            players_turn = 1

    return sars1, sars2


def stats_of_live_against_random(agent, num_games):
    ensemble_wins = 0
    draws = 0
    random_wins = 0
    for i in range(num_games):
        sars1, sars2 = record_eps_greedy_random_training_game(agent, eps=0)
        if sars1[-1][2] == 1:
            ensemble_wins += 1
        elif sars2[-1][2] == 1:
            random_wins += 1
        else:
            draws += 1
    return ensemble_wins, draws, random_wins


def record_competitive_game(agent_1, agent_2, players_start=1, randomness=0.1):
    sars1 = []
    sars2 = []
    players_turn = players_start

    state, _ = Connect4.start_state()
    while not Connect4.evaluate_state(state)[1]:
        if players_turn == 1:
            if len(sars2) > 0:
                sars2[-1][3] = state
            turn = [state, None, None, None]
            action = agent_1.choose_eps_greedy_action(state, Connect4.possible_actions, eps=randomness)
            state = Connect4.take_action(state, action, 1)

            reward, _ = Connect4.evaluate_state(state)
            turn[1] = action
            turn[2] = reward
            sars1.append(turn)

            players_turn = 2

        elif players_turn == 2:
            if len(sars1) > 0:
                sars1[-1][3] = Connect4.reverse_state(state)
            turn = [Connect4.reverse_state(state), None, None, None]
            action = agent_2.choose_eps_greedy_action(Connect4.reverse_state(state), Connect4.possible_actions, eps=randomness)
            state = Connect4.take_action(state, action, 2)

            reward = -1 * Connect4.evaluate_state(state)[0]
            turn[1] = action
            turn[2] = reward
            sars2.append(turn)

            players_turn = 1

    return sars1, sars2

def stats_of_agents_games(agent1, agent2, num_games):
    ensemble_wins = 0
    draws = 0
    random_wins = 0
    for i in range(num_games):
        if i % 2 == 0:
            sars1, sars2 = record_competitive_game(agent1, agent2, players_start=1, randomness=0.1)
        else:
            sars1, sars2 = record_competitive_game(agent1, agent2, players_start=2, randomness=0.1)
        if sars1[-1][2] == 1:
            ensemble_wins += 1
        elif sars2[-1][2] == 1:
            random_wins += 1
        else:
            draws += 1
    return ensemble_wins, draws, random_wins


def play_human(model, player_start=1):
    state = Connect4.start_state()[0]
    players_turn = player_start
    print('Let us begin.')
    print('Player {} to start'.format(players_turn))
    print('')
    Connect4.print_state(state)
    while not Connect4.evaluate_state(state)[1]:
        if players_turn == 1:
            action = model.choose_greedy_action(state, Connect4.possible_actions)
            score = model.action_values_live(tf.convert_to_tensor(np.expand_dims(state, axis=0)))
            print('Value of board: {}'.format(score))
            print('')
            state = Connect4.take_action(state, action, players_turn)
            print('--------------------------------')
            print('')
            Connect4.print_state(state)
            print('')
            print('Computer went for column {}'.format(action + 1))
            print('')

            players_turn = 2
        elif players_turn == 2:

            score = model.action_values_live(tf.convert_to_tensor(np.expand_dims(Connect4.reverse_state(state), axis=0)))
            print('Value of board: {}'.format(score))
            print('')
            action = int(input('Which column would you like to play in?\n')) - 1
            state = Connect4.take_action(state, action, players_turn)
            print('--------------------------------')
            print('')
            Connect4.print_state(state)
            print('')

            players_turn = 1

    print('The game is over.')
