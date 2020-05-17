import ConnectAgent
import ConnectTrain
import ConnectMemory
import Networks
import Connect4
import tensorflow as tf
import tensorflow.keras.optimizers as opt


net = Networks.gpu_net([128, 64, 32], 3, [32, 32], 3, 2, [128])
target_net = Networks.gpu_net([128, 64, 32], 3, [32, 32], 3, 2, [128], zero=True)
mem = ConnectMemory.ValueFunctionMemory2(max_memory=100)
optimizer = opt.Adam()

agent = ConnectAgent.ConnectAgentGPU(net, target_net, mem)


while not agent.memory_full():
    sars1, sars2 = ConnectTrain.record_eps_greedy_self_play_game(agent, eps1=1, eps2=1)
    agent.update_memory(sars1)
    agent.update_memory(sars2)

while True:
    sars1, sars2 = ConnectTrain.record_eps_greedy_self_play_game(agent, eps1=1, eps2=1)
    agent.update_memory(sars1)
    agent.update_memory(sars2)

    agent.train(32, optimizer, Connect4.possible_actions)
print('done')





