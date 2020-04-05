import Env
import Agent
import tensorflow as tf

print(tf.__version__)

player_1 = Agent.IntelligentAgent(50)
player_2 = Agent.RandomAgent()

env = Env.Environment(player_1, player_2)
env.play_game()
