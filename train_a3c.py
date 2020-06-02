import random
from fightingice_env import FightingiceEnv


# 格斗游戏是一个典型的实时动作游戏，玩家在游戏中选择一定的动作，在规
# 定的时间内击败对方角色，赢得胜利。本任务基于FightingICE 格斗游戏平台，
# 以已知的固定bot “MctsAi” 作为游戏对手，利用课堂上讲述的强化学习方法设计
# 游戏AI，通过训练学习得到具有一定智能水平的格斗AI。
# 将最终学到的强化学习AI 与MctsAi 对抗，统计100 局中AI 的胜率，以及
# 每局结束时双方血量差的平均值，以此作为评判强化学习系统的性能优劣。

# 其中.\FighingICE.jar 是格斗游戏的java 程序;
# .\fightingice_env.py 包含了强化学习系统启动格斗游戏的接口程序;
# .\gym_ai.py 包含了强化学习系统控制游戏角色的代码;
# .\data\ai\MctsAi.jar 是基于java 开发的对手bot;
# .\train.py 包含强化学习系统的主要设计框架.

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import collections
import threading
import torch.nn.functional as F

# Hyperparam:
GAMMA = 0.95
LEARNING_RATE = 1e-3
Thread_nums = 8
Score_list = []
Train_step = 0
Epoch_id = 0
PATH = "param\A3C"
Model_PATH = "model\A3C"
Loss_list = []
Policy_coef = 0.03
Critic_coef = 0.5

class AC(nn.Module):
    def __init__(self, input_size, output_size):
        super(AC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, self.output_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs):
        dense_out = self.net(inputs)
        policy_output = F.softmax(self.actor(dense_out), dim = 0)
        critic_output = self.critic(dense_out)



def make_envs(num_of_threads):
    envs_threads = []
    for i in range(num_of_threads):
        envs_threads.append(FightingiceEnv(port=4242))
        print("init: {} env",format(i+1))
    return envs_threads


step_max = 20
score_sum = [0.for _ in range(Thread_nums)]

def runprocess(envs_threads, thread_id, s_t, model, optimizer):

    global Train_step
    global Epoch_id
    global Total_step
    step = 0
    initial = 0
    done_flag = False

    r_ls = []
    s_ls = []
    policy_ls = []
    critic_ls = []

    while step - initial < step_max and done_flag == False:
        s_t = torch.tensor(s_t, dtype = torch.float32)
        policy, critic = model(s_t)
        action_choice = torch.multinomial(policy, 1).numpy()[0]
        p_value = policy[action_choice]
        s_next, reward, done_flag, info = envs_threads[thread_id].step(action_choice)

        s_ls = np.append(s_ls, s_t)
        r_ls = np.append(r_ls, reward)
        # 有问题的
        policy_ls = np.append(policy_ls,p_value.detach().numpy())
        critic_ls = np.append(critic_ls, critic.detach().numpy())

        s_t = s_next
        step += 1
        score_sum[thread_id] += reward

    if done_flag == False:
        r_ls[-1] = critic_ls[-1]
    elif info is not None:
        r_ls[-1] = 0
        Score_list.append(score_sum[thread_id])
        f = open("info\Score_a3c.txt", "a")
        f.write("Epoch: " + str(Epoch_id) + " Score: " + str(score_sum[thread_id]) + "\r\n")
        f.close()
        f = open("info\Hpdiff_a3c.txt", "a")
        f.write("Epoch: " + str(Epoch_id) + " Score: " + str(info[0] - info[1]) + "\r\n")
        f.close()
        score_sum[thread_id] = 0
        Epoch_id += 1
        print("Train Epoch: {} round result: own hp {} vs opp hp {}, you {}".format(Epoch_id + 1, info[0], info[1],'win' if info[0]>info[1] else 'lose'))
        # 模型保存
        if Epoch_id % 50:
            torch.save(model.state_dict(), PATH + str(Epoch_id) + ".ckpt")
            torch.save(model, Model_PATH + str(Epoch_id) + ".pth")
        break
    else:
        # 线程裂开
        pass

    for i in range(2, len(r_ls) - 1):
        r_ls[i] = r_ls[i] + GAMMA * r_ls[i + 1]

    # training
    Train_step += 1
    loss = train_net(model, optimizer, r_ls, critic_ls, policy_ls, Loss_list)
    f = open("info\Loss_a3c.txt", "a")
    f.write("Train step: " + str(Train_step) + " Loss: " + str(loss) + "\r\n")
    f.close()

    return s_t, s_ls, r_ls, critic_ls, policy_ls

def train_net(model, optimizer, return_ls, critic_ls, policy_ls, loss_list):
    critic_ls = torch.tensor(critic_ls, dtype = torch.float32)
    return_ls = torch.tensor(return_ls, dtype = torch.float32)
    policy_ls = torch.tensor(policy_ls, dtype = torch.float32)
    advantage = return_ls - critic_ls

    loss_policy = advantage**2
    loss_value = torch.log(policy_ls)*critic_ls
    loss = -Policy_coef * torch.mean(loss_policy) + Critic_coef * torch.mean(loss_value)
    loss_list.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


Epoch_r = []
Epoch_state = []
Epoch_critic = []
Epoch_policy = []


class actor_thread(threading.Thread):
    def __init__(self, thread_id, s_t, model, optimizer):
        super(actor_thread, self).__init__()
        self.thread_id = thread_id
        self.next_state = s_t
        self.model = model
        self.optimizer = optimizer

    def run(self):
        global Epoch_r
        global Epoch_state
        global Epoch_critic
        global Epoch_policy

        threadLock.acquire()
        self.next_state, state_batch, R_batch, critic_batch, policy_batch = runprocess(self.thread_id, self.next_state, self.model, self.optimizer)
        Epoch_r = np.append(Epoch_r, R_batch)
        Epoch_state = np.append(Epoch_state, state_batch)
        Epoch_critic = np.append(Epoch_critic, critic_batch)
        Epoch_policy = np.append(Epoch_policy, policy_batch)
        threadLock.release()


if __name__ == '__main__':
    envs = make_envs(Thread_nums)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    # env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]

    obs = []
    for i in range(len(envs)):
        obs.append(envs[i].reset(env_args=env_args))

    model = AC(input_size = 144, output_size = 40)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    while True:
        threadLock = threading.Lock()
        threads = []
        # 线程创建
        for i in range(Thread_nums):
            threads.append(actor_thread(i), obs[i], model, optimizer)

        # 线程启动
        for i in range(Thread_nums):
            threads[i].start()

        # 等待线程结束
        for i in range(Thread_nums):
            threads[i].join()

        # 刷新载入状态
        for i in range(Thread_nums):
            obs[i] = threads[i].next_state


