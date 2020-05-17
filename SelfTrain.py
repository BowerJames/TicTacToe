import random

import numpy as np
import tensorflow as tf
import TicTacToeBoard


def record_eps_greedy_random_training_games(ensemble_agent, num_games, eps=0.3):
    sars1 = []
    sars2 = []
    for i in range(num_games):
        epsars1, epsars2 = record_eps_greedy_random_training_game(ensemble_agent, eps=eps)
        sars1 = sars1 + epsars1
        sars2 = sars2 + epsars2

    return sars1, sars2

def record_eps_greedy_random_training_game(ensemble_agent, eps=0.3):
    board = TicTacToeBoard.TicTacToeBoard()
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
            action1 = ensemble_agent.choose_live_eps_greedy_action(state1,
                                                                   TicTacToeBoard.TicTacToeBoard.possible_actions_from_state,
                                                                   eps=eps)

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
            state2 = -1 * board.get_state().astype(int)
            if len(sars2) > 0:
                sars2[-1][2] = reward2
                sars2[-1][3] = state2
            action2 = ensemble_agent.choose_random_action(state2,
                                                          TicTacToeBoard.TicTacToeBoard.possible_actions_from_state)
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

def record_eps_greedy_training_games(ensemble_agent, num_games, eps=0.3):
    sars1 = []
    sars2 = []
    for i in range(num_games):
        epsars1, epsars2 = record_eps_greedy_training_game(ensemble_agent, eps=eps)
        sars1 = sars1 + epsars1
        sars2 = sars2 + epsars2

    return sars1, sars2

def record_eps_greedy_training_game(ensemble_agent, eps=0.3):
    board = TicTacToeBoard.TicTacToeBoard()
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
            action1 = ensemble_agent.choose_live_eps_greedy_action(state1, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state, eps=eps)

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
            state2 = -1 * board.get_state().astype(int)
            if len(sars2) > 0:
                sars2[-1][2] = reward2
                sars2[-1][3] = state2
            action2 = ensemble_agent.choose_policy_action(state2, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state)
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


def record_random_training_game(ensemble_agent):
    board = TicTacToeBoard.TicTacToeBoard()
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
            action1 = ensemble_agent.choose_random_action(state1, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state)

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
            state2 = -1 * board.get_state().astype(int)
            if len(sars2) > 0:
                sars2[-1][2] = reward2
                sars2[-1][3] = state2
            action2 = ensemble_agent.choose_policy_action(state2, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state)
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


def record_random_training_games(ensemble_agent, num_games):
    sars1 = []
    sars2 = []
    for i in range(num_games):
        epsars1, epsars2 = record_random_training_game(ensemble_agent)
        sars1 = sars1 + epsars1
        sars2 = sars2 + epsars2

    return sars1, sars2

def print_policy_game(ensemble_agent):

    board = TicTacToeBoard.TicTacToeBoard()
    players_turn = random.choice([-1, 1])
    print("Player {0} to start\n".format(players_turn, ))
    board.render()
    print("------------------")

    while not board.Termination:
        if players_turn == 1:
            state1 = board.get_state().astype(int)
            action = ensemble_agent.choose_policy_action(state1, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state)
            board.make_move(action, players_turn)
            players_turn *= -1
            board.render()
            print("------------------")
            _ = board.evaluate_board()
        elif players_turn == -1:
            state1 = -1 * board.get_state().astype(int)
            action = ensemble_agent.choose_policy_action(state1, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state)
            board.make_move(action, players_turn)
            players_turn *= -1
            board.render()
            print("------------------")
            _ = board.evaluate_board()

def stats_of_live_against_random(ensemble_agent, num_games):
    ensemble_wins = 0
    draws = 0
    random_wins = 0
    for i in range(num_games):
        sars1, _ = record_eps_greedy_random_training_game(ensemble_agent, eps=0)
        if sars1[-1][2] == 1:
            ensemble_wins += 1
        elif sars1[-1][2] == -1:
            random_wins += 1
        elif sars1[-1][2] == 0:
            draws += 1
    return ensemble_wins, draws, random_wins


def record_eps_greedy_live_vs_live_games(ensemble_agent, number_games, eps1=0.3, eps2=0.3):
    sars1 = []
    sars2 = []
    for i in range(number_games):
        epsars1, epsars2 = record_eps_greedy_live_vs_live_game(ensemble_agent, eps1=eps1, eps2=eps2)
        sars1 = sars1 + epsars1
        sars2 = sars2 + epsars2

    return sars1, sars2


def record_eps_greedy_live_vs_live_game(ensemble_agent, eps1=0.3, eps2=0.3):
    board = TicTacToeBoard.TicTacToeBoard()
    players_turn = 1
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
            action1 = ensemble_agent.choose_live_eps_greedy_action(state1,
                                                          TicTacToeBoard.TicTacToeBoard.possible_actions_from_state,
                                                            eps=eps1)

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
            state2 = -1 * board.get_state().astype(int)
            if len(sars2) > 0:
                sars2[-1][2] = reward2
                sars2[-1][3] = state2
            action2 = ensemble_agent.choose_live_eps_greedy_action(state2,
                                                          TicTacToeBoard.TicTacToeBoard.possible_actions_from_state,
                                                                   eps=eps2)
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


def print_live_game(ensemble_agent):

    board = TicTacToeBoard.TicTacToeBoard()
    players_turn = random.choice([-1, 1])
    print("Player {0} to start\n".format(players_turn, ))
    board.render()
    print("------------------")

    while not board.Termination:
        if players_turn == 1:
            state1 = board.get_state().astype(int)
            action = ensemble_agent.choose_live_eps_greedy_action(state1, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state, eps=0)
            board.make_move(action, players_turn)
            players_turn *= -1
            board.render()
            print("------------------")
            _ = board.evaluate_board()
        elif players_turn == -1:
            state1 = -1 * board.get_state().astype(int)
            action = ensemble_agent.choose_random_action(state1, TicTacToeBoard.TicTacToeBoard.possible_actions_from_state)
            board.make_move(action, players_turn)
            players_turn *= -1
            board.render()
            print("------------------")
            _ = board.evaluate_board()


def record_eps_game_with_knowledge(ensemble_agent, eps1=0.3, eps2=0.3):
    board = TicTacToeBoard.TicTacToeBoard()
    players_turn = 1
    sars1 = []
    sars2 = []
    reward1 = None
    reward2 = None
    while not board.Termination:
        if players_turn == 1:
            state1 = board.get_state().astype(int)
            if len(sars2) > 0:
                sars2[-1][3] = state1
            action1 = ensemble_agent.choose_live_eps_greedy_action(state1,
                                                                   TicTacToeBoard.TicTacToeBoard.possible_actions_from_state,
                                                                   eps=eps1)

            board.make_move(action1, 1)
            reward1 = board.evaluate_board()
            sars1.append([state1, action1, reward1, None])
            players_turn *= -1

        elif players_turn == -1:
            state2 = -1 * board.get_state().astype(int)
            if len(sars1) > 0:
                sars1[-1][3] = state2
            action2 = ensemble_agent.choose_live_eps_greedy_action(state2,
                                                                   TicTacToeBoard.TicTacToeBoard.possible_actions_from_state,
                                                                   eps=eps2)
            board.make_move(action2, -1)
            reward2 = -1 * board.evaluate_board()
            sars2.append([state2, action2, reward2, None])

            players_turn *= -1

    return sars1, sars2


def record_eps_games_with_knowledge(ensemble_agent, num_games, eps1=0.3, eps2=0.3):
    sars1 = []
    sars2 = []
    for i in range(num_games):
        epsars1, epsars2 = record_eps_game_with_knowledge(ensemble_agent, eps1=eps1, eps2=eps2)
        sars1 = sars1 + epsars1
        sars2 = sars2 + epsars2

    return sars1, sars2