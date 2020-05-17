import numpy as np


def start_state():
    board = np.zeros(shape=(6, 7, 3))
    board[:, :, 0] = 1
    return board, False


def possible_actions(state):
    if np.array_equal(state, np.zeros((6, 7, 3))):
        return [0]
    action = []
    for i in range(7):
        if 1 in state[:, i, 0]:
            action.append(i)
    return action


def take_action(state, action, players_turn):
    new_state = np.copy(state)
    spot = np.zeros((3,))
    spot[players_turn] = 1
    column = new_state[:, action, 0]
    row_num = 0
    while row_num < 6:
        if column[row_num] == 1:
            break
        row_num += 1
    new_state[(row_num, action)] = spot

    return new_state

def evaluate_state(state):
    empty_frame = state[:, :, 0]
    player_1_frame = state[:, :, 1]
    player_2_frame = state[:, :, 2]
    for i in range(6):
        for j in range(7):
            if win_from(player_1_frame, (i, j)):
                return 1, True
            if win_from(player_2_frame, (i, j)):
                return -1, True
    if full(empty_frame):
        return 0, True

    return 0, False

def win_from(frame, index):
    y = index[0]
    x = index[1]
    if y < 3:
        if np.sum(frame[y : y + 4, x]) == 4:
            return True
    if x < 4:
        if np.sum(frame[y, x : x + 4]) == 4:
            return True
    if y < 3 and x < 4:
        if sum([frame[y, x], frame[y+1, x+1], frame[y+2, x+2], frame[y+3, x+3]]) == 4:
            return True
    if y > 2 and x < 4:
        if sum([frame[y, x], frame[y-1, x+1], frame[y-2, x + 2], frame[y-3, x+3]]) == 4:
            return True
    return False


def full(empty_frame):
    return np.sum(empty_frame) == 0


def print_state(state):
    for i in range(6):
        string = '|'
        row = state[5-i, :, :]

        for j in range(7):
            spot = row[j, :]
            if spot[0] == 1:
                string = string + ' '
            elif spot[1] == 1:
                string = string + 'o'
            elif spot[2] == 1:
                string = string + 'x'
            string = string + '|'

        print(string)


def reverse_state(state):

    rev_state = np.zeros(shape=(6, 7, 3))
    rev_state[:, :, 0] = state[:, :, 0]
    rev_state[:, :, 1] = state[:, :, 2]
    rev_state[:, :, 2] = state[:, :, 1]

    return rev_state






