import random
from fightingice_env import FightingiceEnv
import matplotlib.pyplot as plt

# 格斗游戏是一个典型的实时动作游戏，玩家在游戏中选择一定的动作，在规
# 定的时间内击败对方角色，赢得胜利。本任务基于FightingICE 格斗游戏平台，
# 以已知的固定bot “MctsAi” 作为游戏对手，利用课堂上讲述的强化学习方法设计
# 游戏AI，通过训练学习得到具有一定智能水平的格斗AI。
# TODO:将最终学到的强化学习AI 与MctsAi 对抗，统计100 局中AI 的胜率，以及
# TODO:每局结束时双方血量差的平均值，以此作为评判强化学习系统的性能优劣。

#  其中.\FighingICE.jar 是格斗游戏的java 程序;
# .\fightingice_env.py 包含了强化学习系统启动格斗游戏的接口程序;
# .\gym_ai.py 包含了强化学习系统控制游戏角色的代码;
# .\data\ai\MctsAi.jar 是基于java 开发的对手bot;
# .\train.py 包含强化学习系统的主要设计框架.

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import collections
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size, mem_len):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory = collections.deque(maxlen = mem_len)
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Dueling 架构
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, self.output_size)

    def forward(self, input):
        net_output = self.net(input)
        v = self.V(net_output)
        advantage = self.A(net_output)
        advantage = advantage - torch.mean(advantage)
        q_value = v + advantage
        return q_value

    def sample_action(self, inputs, epsilon):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        inputs = inputs.unsqueeze(0)
        q_value = self(inputs)
        seed = np.random.rand()
        if seed > epsilon:
            action_choice = int(torch.argmax(q_value))
        else:
            action_choice = random.choice(range(self.output_size))
        return action_choice

    def save_trans(self, transition):
        self.memory.append(transition)

    def sample_memory(self, batch_size):
        s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], []
        trans_batch = random.sample(self.memory, batch_size)
        for trans in trans_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        return torch.tensor(s_ls,dtype=torch.float32),\
            torch.tensor(a_ls,dtype=torch.int64),\
            torch.tensor(r_ls,dtype=torch.float32),\
            torch.tensor(s_next_ls,dtype=torch.float32),\
            torch.tensor(done_flag_ls,dtype=torch.float32)


def train_net(Q_net, Q_target, optimizer, losses, loss_list, replay_time, gamma, batch_size):
    s, a, r, s_next, done_flag = Q_net.sample_memory(batch_size)
    # for i in range(replay_time):
    q_value = Q_net(s)
    a = torch.LongTensor(a)
    q_value = torch.gather(q_value, 1, a)

    q_t = Q_net(s_next)
    a_index = torch.argmax(q_t, 1)
    a_index = a_index.reshape((a_index.shape[0], 1))
    # print(a.size())
    # print(a_index.shape)
    q_target = Q_target(s_next)
    q_target = torch.gather(q_target, 1, a_index)
    q_target = r + gamma * q_target * done_flag

    loss = losses(q_target, q_value)
    loss_list.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_curse(target_list, loss_list):
    figure1 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(target_list)):
        X.append(i)
    plt.plot(X,target_list,'-r')
    plt.xlabel('epoch')
    plt.ylabel('score')

    figure2 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(loss_list)):
        X.append(i)
    plt.plot(X,loss_list,'-b')
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.show()


# Hyperparam
max_epoch = 10000
path = "param\DDQN"
model_path = "model\DDQN"
R_info = "info\Reward_DDQN"
L_info = "info\Loss_DDQN"
hp_info = "info\Hp_diff"
# 创建模型和参数的保存目录
ensure_dir(path)
ensure_dir(model_path)
ensure_dir(R_info)
LOAD_KEY = True
learning_rate = 1e-3
max_steps = 300
replay_time = 1
epsilon = 0.01
gamma = 0.95
step_count = 0
train_flag = False
mem_len = 30000
train_begin = 20000 #4000
batch_size = 32
Hp_diff = []
win_count = 0

if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    # env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "4", "--mute"]

    Q_net = DQN(input_size = 144, output_size = 40, mem_len = mem_len)
    Q_target = DQN(input_size = 144, output_size = 40, mem_len = mem_len)
    Q_target.load_state_dict(Q_net.state_dict())

    if LOAD_KEY:
        epoch_model_param = 960
        Q_net = torch.load(model_path + str(epoch_model_param) + ".pth")
        Q_target.load_state_dict(Q_net.state_dict())
        # score_m = np.load(R_info + str(690) + ".npy")
        # score_list = score_m.tolist()
        # loss_m = np.load(L_info + str(690) + ".npy")
        # loss_list = loss_m.tolist()
        print("Load Weights and param/info!")

    optimizer = optim.Adam(Q_net.parameters(), lr = learning_rate)
    losses = nn.MSELoss()

    loss_list = []
    score_list = []

    for epo_i in range(960, max_epoch):
        obs = env.reset(env_args=env_args)
        # reward, done, info = 0, False, None
        score = 0.
        epsilon = max(0.01, epsilon*0.999)
        for step in range(max_steps):
            step_count += 1
            action_choice = Q_net.sample_action(obs, epsilon)
            new_obs, reward, done, info = env.step(action_choice)
            if done:
                done_flag = 0
                new_obs = [0. for i in range(len(obs))]
            else:
                done_flag = 1
            Q_net.save_trans((obs, action_choice, reward, new_obs, done_flag))
            score += reward
            # train
            if step_count > train_begin:
                train_flag = True
                train_net(Q_net, Q_target, optimizer, losses, loss_list, 1, gamma, batch_size)
            # target copy online net
            if step_count % 3000 == 0 and train_flag == True:
                Q_target.load_state_dict(Q_net.state_dict())

            obs = new_obs

            # infomation print
            if done or step + 1 == max_steps:
                score_list.append(score)

                # if (epo_i+1) % 30 == 0 and train_flag == True:
                #     print("Log information and save weights/models...")
                #     torch.save(Q_net.state_dict(), path + str(epo_i+1) + ".ckpt")
                #     torch.save(Q_net, model_path + str(epo_i+1) + ".pth")
                #     score_np = np.array(score_list)
                #     loss_np = np.array(loss_list)
                #     np.save(R_info + str(epo_i + 1) + ".npy", score_np)
                #     np.save(L_info + str(epo_i + 1) + ".npy", loss_np)

                if done:
                    if info is not None:
                        if info[0] > info[1]:
                            win_count += 1      # 胜率计算
                            f = open(hp_info + ".txt", "a")
                            f.write("Epoch: " + str(epo_i + 1) + " Hp_diff: " + str(info[0] - info[1]) + "\r")
                            f.close()

                if info is not None:
                    # f = open(hp_info + ".txt", "a")
                    # f.write("Epoch: " + str(epo_i + 1) + " Hp_diff: " + str(info[0] - info[1]) + "\r\n")
                    # f.close()
                    # Hp_diff.append(info[0] - info[1])
                    print("Epoch: {}  round result: own hp {} vs opp hp {}, you {}   training: {}    epsilon: {}  done: {} step_count: {}".format(epo_i, info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose', train_flag, epsilon, done, step_count))
                else:
                    # java terminates unexpectedly
                    pass
                break


    plot_curse(score_list, loss_list)
    print("finish training")


# 示例程序
# if __name__ == '__main__':
#     env = FightingiceEnv(port=4242)
#     # for windows user, port parameter is necessary because port_for library does not work in windows
#     # for linux user, you can omit port parameter, just let env = FightingiceEnv()
#
#     env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]
#     # this mode let two players have infinite hp, their hp in round can be negative
#     # you can close the window display functional by using the following mode
#     #env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]
#
#     while True:
#         obs = env.reset(env_args=env_args)
#         reward, done, info = 0, False, None
#
#         while not done:
#             act = random.randint(0, 10)
#             # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
#             new_obs, reward, done, info = env.step(act)
#
#             if not done:
#                 # TODO: (main part) learn with data (obs, act, reward, new_obs)
#                 # suggested discount factor value: gamma in [0.9, 0.95]
#                 pass
#             elif info is not None:
#                 print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
#                                                                             'win' if info[0]>info[1] else 'lose'))
#             else:
#                 # java terminates unexpectedly
#                 pass
#
#     print("finish training")