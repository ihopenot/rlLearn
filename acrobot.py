import sys
is_upload = False
if len(sys.argv) > 1 and sys.argv[1] == 'upload':
    is_upload = True

if is_upload:
    from comet_ml import Experiment
    from comet_ml.integration.gymnasium import CometLogger
    experiment = Experiment(
    project_name="general",
    workspace="ihopenot"
    )
          
import gymnasium as gym
from alg.dqn import *
from tqdm import tqdm

env = gym.make("Acrobot-v1", render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, './video', episode_trigger=lambda x: x % 100 == 0)
if is_upload:
    env = CometLogger(env, experiment)
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.03
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = ReplayBuffer(buffer_size)
# state_dim = env.observation_space.shape[0]
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(DUEL_QNet, state_dim, action_dim, lr, gamma, epsilon,
            target_update, device, DQN.DOUBLE_DQN)

return_list = []
done = False
state, _ = env.reset()
episode_return = 0
for i in tqdm(range(num_episodes)):
    env.reset()
    episode_return = 0
    while True:
        action = agent.take_action(state)
        next_state, reward, done, _, info = env.step(action)
        # env.render()

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        episode_return += reward
        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            agent.update(transition_dict)

        if done:
            break

    return_list.append(episode_return)
    if i % 100 == 0:
        print(f'episode: {i}, return: {episode_return}')

env.close()
