import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import collections
import gym
import game.wuziqi

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(QNet, self).__init__()
        dim = 1
        for i in state_dim:
            dim *= i
        self.fc1 = torch.nn.Linear(dim, 256)
        # self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(256, action_dim)

    def forward(self, x):
        x = x.view(-1, 3*8*8)
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DUEL_QNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(DUEL_QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fcA = torch.nn.Linear(128, action_dim)
        self.fcV = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        A = self.fcA(x)
        V = self.fcV(x)
        Q = V + A - A.mean()
        return Q

# input (1 * 2 * 8 * 8)
class QNet2(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super(QNet2, self).__init__()
        # self.conv1 = torch.nn.Conv2d(2, 8, 3, padding=1)
        # self.conv2 = torch.nn.Conv2d(4, 8, 3, padding=1)
        # self.conv3 = torch.nn.Conv2d(8, 16, 3, padding=1)
        # (1 * 16 * 2 * 2)
        self.fc1 = torch.nn.Linear(2*8*8, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_dim)

    def forward(self, x):
        # print(x.shape)
        # x = torch.relu(self.conv1(x))
        # x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(-1, 2*8*8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

class WZQNet(torch.nn.Module):
    def __init__(self, w, h) -> None:
        super(WZQNet, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError

    def get_state(env):
        raise NotImplementedError
    
    def rev_state(state):
        raise NotImplementedError
    

# 8 * 8 white
# 8 * 8 black
# 8 * 8 player
class WZQNetL(WZQNet):
    def __init__(self, w, h) -> None:
        super(WZQNetL, self).__init__(w, h)
        self.fc1 = torch.nn.Linear(3*w*h, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, w*h)
    
    def forward(self, x):
        x = x.view(-1, 3*8*8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    
    def get_state(env: game.wuziqi.Env) -> np.ndarray:
        ret_state = np.zeros((3, env.board.width, env.board.height))
        if env.board.states:
            moves, players = np.array(list(zip(*env.board.states.items())))
            move_1 = moves[players == 1]
            move_2 = moves[players != 1]
            ret_state[0][move_1 // env.board.width,
                            move_1 % env.board.height] = 1.0
            ret_state[1][move_2 // env.board.width,
                            move_2 % env.board.height] = 1.0
            # if env.board.current_player % 2 == 0:
            #     ret_state[2][:, :] = 1.0

        # print(square_state)
        return ret_state
    
    def rev_state(state: np.ndarray) -> np.ndarray:
        ret_state = np.zeros(state.shape)
        # ret_state[0] = state[1]
        # ret_state[1] = state[0]
        ret_state[0] = state[0]
        ret_state[1] = state[1]
        # if state[2][0][0] > 0.1:
        #     ret_state[2][:, :] = 0
        # else:
        #     ret_state[2][:, :] = 1
        return ret_state

class DQN:
    DOUBLE_DQN = 1
    DEULING_DQN = 2
    def __init__(
        self,
        nettype: torch.nn.Module,
        state_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        target_update,
        device,
        flags
    ) -> None:
        self.action_dim = action_dim
        self.nettype = nettype
        self.flags = flags
        # self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        # self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.q_net = nettype(state_dim, action_dim).to(device)
        self.target_q_net = nettype(state_dim, action_dim).to(device)

        # TODO: 优化器是啥
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state, possible):
        if np.random.random() < self.epsilon:
            # action = np.random.choice(self.action_dim)
            action = np.random.choice(possible)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.q_net(state)
            # choose argmax in possible moves
            # action = np.argmax(q_values[0][possible].detach().cpu().numpy())

            # print(q_values)
            action = torch.argmax(q_values[0][possible]).item()
            action = possible[action]
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # q_values = self.q_net(states).gather(1, actions)  # Q值
        qtensor = self.q_net(states)
        # print(states.shape, actions.shape)
        q_values = torch.gather(qtensor, 1, actions)

        # 下个状态的最大Q值
        if self.flags & DQN.DOUBLE_DQN:
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        # print(dqn_loss)

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))
        if reward != 0:
            for _ in range(20):
                self.buffer.append((state, action, reward, next_state, done))
            

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


# class DQNTrain:
#     def __init__(self, dqn: DQN) -> None:
#         self.dqn = dqn
#         self.reply_buffer = ReplayBuffer(10000)
#         self.return_list = []
#         self.min_size = 500
#         self.batch_size = 64

#     def run(self, episodes):
#         for i in range(10):
#             with tqdm(total=episodes // 10) as pbar:
#                 for i_ep in range(episodes // 10):
#                     er = 0
#                     board = Board()
#                     board.init_board()
#                     state = board.current_state()
#                     done = False
#                     while not done:
#                         action = self.dqn.take_action(state)
#                         if action not in board.availables:
#                             nxt_state = state
#                             reward = -10
#                             done = True
#                         else:
#                             board.do_move(action)
#                             nxt_state = board.current_state()
#                             end, winner = board.game_end()
#                             if end:
#                                 if winner == 1:
#                                     reward = 10
#                                 else:
#                                     reward = -10
#                             done = end

#                         self.reply_buffer.add(state, action, nxt_state, reward, done)

#                         state = nxt_state
#                         er += reward

#                         # TODO: unimplemented
#                         if self.reply_buffer.size() > self.min_size:
#                             bs, ba, br, bns, bd = self.reply_buffer.sample(
#                                 self.batch_size
#                             )
#                             transition_dict = {
#                                 "states": bs,
#                                 "actions": ba,
#                                 "next_states": bns,
#                                 "rewards": br,
#                                 "dones": bd,
#                             }
#                             self.dqn.update(transition_dict)

#                     self.return_list.append(er)
#                     pbar.update(1)


# # TODO: unimplemented
# dqn = DQN(64, 64, 128, 2e-3, 0.98, 0.01, 10, device)
# trainer = DQNTrain(dqn)
# trainer.run(100000)
