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
from torch.distributions import Categorical


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class PPO(nn.Module):
    def __init__(self, input_size, out_size):
        super(PPO, self).__init__()
        self.input_size = input_size
        self.output_size = out_size
        self.mem = []
        # net
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, self.output_size)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE)

    def get_policy(self, inputs, dim):
        fc  = self.net(inputs)
        policy = self.policy(fc)
        policy = F.softmax(policy, dim = dim)
        return policy

    def get_critic(self, inputs):
        fc = self.net(inputs)
        critic = self.critic(fc)
        return critic

    def save_trans(self, trans):
        self.mem.append(trans)

    def package_trans(self):
        s_ls, a_ls, r_ls, s_next_ls, a_prob_ls, done_flag_ls = [], [], [], [], [], []
        for trans in self.mem:
            s, a, r, s_next, a_prob, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            a_prob_ls.append([a_prob])
            done_flag_ls.append([done_flag])
        s, a, r, s_next, a_prob, done_flag = torch.tensor(s_ls, dtype = torch.float32),\
                                                torch.tensor(a_ls, dtype = torch.int64),\
                                                torch.tensor(r_ls, dtype = torch.float32),\
                                                torch.tensor(s_next_ls, dtype = torch.float32),\
                                                torch.tensor(a_prob_ls, dtype = torch.float32),\
                                                torch.tensor(done_flag_ls, dtype = torch.float32)
        self.mem = []
        return s, a, r, s_next, a_prob, done_flag



def train(model, loss_fn, loss_list, score_list):
    s, a, r, s_next, a_prob, done_flag = model.package_trans()
    for i in range(K_EPOCH):
        td_target = r + GAMMA*model.get_critic(s_next)*done_flag
        td_error = td_target - model.get_critic(s)
        td_error = td_error.detach().numpy()

        advantage_ls = []
        advantage = 0.
        for error in td_error[::-1]:
            advantage = GAMMA * LAMBDA * advantage + error[0]
            advantage_ls.append([advantage])
        advantage_ls.reverse()
        advantage = torch.tensor(advantage_ls, dtype = torch.float32)

        policy = model.get_policy(s, 1)
        policy = policy.gather(1, a)
        ratio = torch.exp(torch.log(policy) - torch.log(a_prob)) # 重要性采样比率？

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP)*advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.get_critic(s), td_target.detach())

        loss_list.append(loss.mean())
        model.optimizer.zero_grad()
        loss.mean().backward()
        model.optimizer.step()


# 保存路径
path = "param\PPO"
model_path = "model\PPO"
R_info = "info\Reward_PPO"
L_info = "info\Loss_PPO"
hp_info = "info\Hp_diff_PPO"
# 创建模型和参数的保存目录
ensure_dir(path)
ensure_dir(model_path)
ensure_dir(R_info)

# Hyperparam
MAX_EPOCH = 10000
LOAD_KEY = True
LEARNING_RATE = 0.0008
max_steps = 20
GAMMA = 0.95
LAMBDA = 0.95
EPS_CLIP = 0.1
K_EPOCH = 3
T_HORIZON = 20
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

    model = PPO(144, 40)

    loss_list = []
    score_list = []
    init_epoch = 0

    if LOAD_KEY:
        epoch_model_param = 1110
        init_epoch = epoch_model_param
        model = torch.load(model_path + str(epoch_model_param) + ".pth")
        score_m = np.load(R_info + str(epoch_model_param) + ".npy")
        score_list = score_m.tolist()
        loss_m = np.load(L_info + str(epoch_model_param) + ".npy")
        loss_list = loss_m.tolist()
        print("Load Weights and param/info!")


    for epo_i in range(init_epoch, MAX_EPOCH):
        obs = env.reset(env_args=env_args)
        score = 0.
        done = False
        while not done:
            for step in range(max_steps):
                step_count += 1
                a_prob = model.get_policy(torch.from_numpy(obs).float(), 0)
                # m = Categorical(a_prob)
                # a = m.sample().item()
                a = int(torch.argmax(a_prob).item())
                obs_next, r, done, info = env.step(a)

                done_flag = 1.0 if not done else 0
                if done:
                    obs_next = [0. for i in range(len(obs))]
                model.save_trans((obs, a, r, obs_next, a_prob[a].item(), done_flag))

                obs = obs_next
                score += r

                # infomation print and save
                if done or ( step + 1 ) == max_steps:
                    # 展示不再训练
                    # train(model, nn.MSELoss(), loss_list, score_list=None)
                    # train_flag = True
                    score_list.append(score)

                    if done:
                        # if (epo_i+1) % 30 == 0:
                        #     print("Log information and save weights/models...")
                        #     torch.save(model.state_dict(), path + str(epo_i+1) + ".ckpt")
                        #     torch.save(model, model_path + str(epo_i+1) + ".pth")
                        #     score_np = np.array(score_list)
                        #     loss_np = np.array(loss_list)
                        #     np.save(R_info + str(epo_i + 1) + ".npy", score_np)
                        #     np.save(L_info + str(epo_i + 1) + ".npy", loss_np)

                        if info is not None:
                            f = open(hp_info + ".txt", "a")
                            f.write("Epoch: " + str(epo_i + 1) + " Hp_diff: " + str(info[0] - info[1]) + "\r")
                            f.close()
                            Hp_diff.append(info[0] - info[1])
                            print("Epoch: {}  round result: own hp {} vs opp hp {}, you {}   training: {}  done: {} step_count: {} ".format(epo_i, info[0], info[1],
                                                                                    'win' if info[0]>info[1] else 'lose', train_flag, done, step_count))
                        else:
                            # java terminates unexpectedly
                            pass
                        break

    print("finish training")
