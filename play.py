from alg.dqn import *
# from game.wuziqi import Env
from game.gomoku import GomokuEnv
from tqdm import tqdm

def playwith(model_name, board_size, win_len):
    env = GomokuEnv("white", "random", "numpy3c", "raise", board_size, win_len)
    lr = 2e-5
    gamma = 0.98
    epsilon = 0
    target_update = 10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = DQN(QNet, state_dim, action_dim, lr, gamma, epsilon,
                target_update, device, DQN.DEULING_DQN & DQN.DOUBLE_DQN)

    def aiplay(cur, prev, prev_a):
        return agent.take_action(cur, possible=GomokuEnv.get_possible_actions(cur))
    env.opponent = aiplay

    agent.q_net = torch.load(model_name)

    done = False
    env.reset()

    while not done:
        env.render()
        x, y = map(int, input().split(","))
        action = x*board_size + y - board_size - 1
        # print(action)
        next_state, reward, done, info = env.step(action)


playwith("out/connect-5-8_1800_0.pth", 8, 5)