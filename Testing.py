import copy

import Env
import Agent
import tensorflow as tf

print(tf.__version__)

player_1 = Agent.IntelligentAgent(10)
player_2 = Agent.RandomAgent()

env = Env.Environment(player_1, player_2)
print('For Epoch 0')
print('Total Reward %d' % env.calc_tot_rewards(number_games=500))

for i in range(100):
    env.train(num_updates=1, episodes_per_batch=500)
    print('')
    print('For Epoch %d' % i)
    env.play_best_game()
    print('Total Reward %d' % env.calc_tot_rewards(number_games=500))
