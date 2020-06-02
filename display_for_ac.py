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

class Actor_critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor_critic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, self.output_size)
        self.critic = nn.Linear(128, 1)
        self.mem_trans = []

    def forward(self, inputs):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        fc = self.net(inputs)
        policy = F.softmax(self.actor(fc), dim=0)
        critic = self.critic(fc)

        return policy, critic

    def choose_action(self, inputs, epsilon):
        policy, critic = self(inputs)
        # print(sum(policy), sum(inputs), critic)
        choice = torch.multinomial(policy, 1)
        return int(choice[0])

    def save_trans(self, trans):
        self.mem_trans.append(trans)


def train_net(model, optimizer, losses, loss_list, gamma):
    loss = 0.
    for trans in model.mem_trans:
        s, a, r, s_next, done_flag = trans
        policy, critic = model(s)
        critic_next = r + gamma*model(s_next)[1]*done_flag
        advantage = critic_next - critic
        policy_loss = torch.log(policy[a]) * advantage
        critic_loss = advantage**2
        loss += -policy_coef * policy_loss + critic_coef * critic_loss

    loss_list.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.mem_trans = []

    # for num, trans in enumerate(trans_list):
    #     s, a, r, s_next, done_flag = trans
    #     policy, critic = model(s)
    #     advantage = r + GAMMA*model(s_next)[1]*done_flag - critic
    #     #
    #     # advantage = return_list[num] - critic
    #     loss_critic = advantage**2
    #     loss_policy = torch.log(policy[a])*advantage
    #     loss = -Policy_coef * loss_policy + Critic_coef * loss_critic
    #
    #     loss_list.append(loss)
    #     optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     optimizer.step()



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

# 保存路径
path = "param\AC"
model_path = "model\AC"

# 创建模型和参数的保存目录
ensure_dir(path)
ensure_dir(model_path)

# Hyperparam
max_epoch = 1000
policy_coef = 0.6
critic_coef = 0.5
LOAD_KEY = True
learning_rate = 1e-3
max_steps = 500
epsilon = 0.01
gamma = 0.95
step_count = 0
train_flag = False
Hp_diff = []

if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    # env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "4", "--mute"]

    model = Actor_critic(144, 40)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = nn.MSELoss()

    loss_list = []
    score_list = []
    init_epoch = 0

    if LOAD_KEY:
        epoch_model_param = 210
        init_epoch = epoch_model_param
        model = torch.load(model_path + str(epoch_model_param) + ".pth")
        # score_m = np.load(R_info + str(690) + ".npy")
        # score_list = score_m.tolist()
        # loss_m = np.load(L_info + str(690) + ".npy")
        # loss_list = loss_m.tolist()
        print("Load Weights and param/info!")


    for epo_i in range(init_epoch, max_epoch):
        obs = env.reset(env_args=env_args)
        # reward, done, info = 0, False, None
        score = 0.
        # epsilon = max(0.01, epsilon*0.999)
        for step in range(max_steps):
            step_count += 1
            action_choice = model.choose_action(obs, epsilon)
            new_obs, reward, done, info = env.step(action_choice)
            if done:
                done_flag = 0
                new_obs = [0. for i in range(len(obs))]
            else:
                done_flag = 1
            model.save_trans((obs, action_choice, reward, new_obs, done_flag))
            score += reward

            obs = new_obs

            # infomation print and save
            if done or ( step + 1 ) == max_steps:
                # 训练
                # train_net(model, optimizer, losses, loss_list, gamma)
                train_flag = True
                score_list.append(score)

                # if (epo_i+1) % 30 == 0:
                #     print("Log information and save weights/models...")
                #     torch.save(model.state_dict(), path + str(epo_i+1) + ".ckpt")
                #     torch.save(model, model_path + str(epo_i+1) + ".pth")
                #     score_np = np.array(score_list)
                #     loss_np = np.array(loss_list)
                #     np.save(R_info + str(epo_i + 1) + ".npy", score_np)
                #     np.save(L_info + str(epo_i + 1) + ".npy", loss_np)

                if info is not None:
                    # f = open(hp_info + ".txt", "a")
                    # f.write("Epoch: " + str(epo_i + 1) + " Hp_diff: " + str(info[0] - info[1]) + "\r")
                    # f.close()
                    Hp_diff.append(info[0] - info[1])
                    print("Epoch: {}  round result: own hp {} vs opp hp {}, you {}   training: {}    epsilon: {}  done: {} step_count: {}".format(epo_i, info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose', train_flag, epsilon, done, step_count))
                else:
                    # java terminates unexpectedly
                    pass
                break

    plot_curse(score_list, loss_list)
    print("finish training")
