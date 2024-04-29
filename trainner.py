from alg.dqn import *
# from game.wuziqi import Env
from game.gomoku import GomokuEnv
from tqdm import tqdm
from typing import List

env = GomokuEnv("black", "random", "numpy3c", "raise", 8, 5)


lr = 2e-4
num_episodes = 20000
gamma = 0.98
epsilon = 0.03
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = [ReplayBuffer(buffer_size) for _ in range(2)]
state_dim = env.observation_space.shape
action_dim = env.action_space.n
agents : List[DQN] = [None, None]
agents[0] = DQN(QNet, state_dim, action_dim, lr, gamma, epsilon,
            target_update, device, DQN.DEULING_DQN & DQN.DOUBLE_DQN)

agents[1] = DQN(QNet, state_dim, action_dim, lr, gamma, epsilon,
            target_update, device, DQN.DEULING_DQN & DQN.DOUBLE_DQN)

return_list = []
episode_return = [0, 0, 0]
for i in tqdm(range(num_episodes)):
    done = [False, False]
    state = env.reset()
    # state = agent.nettype.get_state(env)

    last = [None, None]
    laction = [None, None]
    rewards = [0, 0]
    player = 0

    while not done[0] or not done[1]:
        if done[player^1]:
            rewards[player] = 0 if rewards[player^1] == 0 else -1
            replay_buffer[player].add(last[player], laction[player], rewards[player], state, True)
            done[player] = True

        else:
            if last[player] is not None:
                replay_buffer[player].add(last[player], laction[player], rewards[player], state, False)

            action = agents[player].take_action(state, possible=GomokuEnv.get_possible_actions(state))
            next_state, rewards[player], done[player], info = env.step_single(action, player)
            if done[player]:
                replay_buffer[player].add(state, action, rewards[player], next_state, True)

            last[player] = state
            laction[player] = action
            state = next_state

        if replay_buffer[player].size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer[player].sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            agents[player].update(transition_dict)

        player ^= 1
    
    # print(rewards)
    episode_return[rewards[0] + 1] += 1
    # return_list.append(episode_return)
    if i % 100 == 0:
        print(f'episode: {i}, return: {episode_return}')
    if i % 1000 == 0:
        torch.save(agents[0].q_net, f'out/connect-5-8_{i}_0.pth')
        torch.save(agents[1].q_net, f'out/connect-5-8_{i}_1.pth')


