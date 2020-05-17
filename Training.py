
import ConnectAgent
import ConnectTrain
import ConnectMemory
import Networks
import Connect4
import tensorflow.keras.models as tkm
import tensorflow.keras.optimizers as opt
import tensorflow.keras.losses as losses
import tensorflow as tf


net = Networks.res_model([128, 64, 32], 3, [32, 32])
net.summary()
optimizer = opt.Adam()
var = net.trainable_variables
mem = ConnectMemory.ValueFunctionMemory2(max_memory=10000)
target_net = Networks.state_to_action_net_zero((6, 7, 3), [32, 32, 8])

agent = ConnectAgent.StateToActionAgent(qnet=net, memory=mem, target_net=target_net)
#agent = ConnectAgent.StateToActionAgent.load('./Connect8')

old_agent = ConnectAgent.ConnectAgent.load('./SecondConnect')
'''
ConnectTrain.play_human(agent)
'''

while True:

    agent_wins, draws, random_wins = ConnectTrain.stats_of_agents_games(agent, old_agent, 100)
    print('')
    print('Agent1 wins: ' + str(agent_wins))
    print('Draws: ' + str(draws))
    print('Agent2 Wins: ' + str(random_wins))
    print('')

    agent_wins, draws, random_wins = ConnectTrain.stats_of_live_against_random(agent, 100)
    print('')
    print('Agent1 wins: ' + str(agent_wins))
    print('Draws: ' + str(draws))
    print('Random Wins: ' + str(random_wins))
    print('')
    while not agent.memory_full():
        sars1, sars2 = ConnectTrain.record_eps_greedy_self_play_game(agent, eps1=1, eps2=1)
        agent.update_memory(sars1)
        agent.update_memory(sars2)


    for _ in range(500):
        sars1, sars2 = ConnectTrain.record_eps_greedy_self_play_game(agent, eps1=1, eps2=1)
        agent.update_memory(sars1)
        agent.update_memory(sars2)
        agent.train2(10, optimizer, 128, Connect4.possible_actions)
    #agent.update_target_weights()
    agent.save('./Connect9')
