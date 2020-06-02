import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym
from collections import deque
from fightingice_env import FightingiceEnv
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
from torch.distributions import Categorical


class policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, 1)

    def act(self, input):
        probs = self.forward(input)
        dist = Categorical(probs)
        action = dist.sample()
        action = action.detach().item()
        return action


if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    model = torch.load('model/icm_gae_ppo_policy.pkl')
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "10", "--mute"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    #env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]

    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None

        while not done:
            act = model.act(torch.FloatTensor(np.expand_dims(obs, 0)))
            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            obs, reward, done, info = env.step(act)

            if not done:
                # TODO: (main part) learn with data (obs, act, reward, new_obs)
                # suggested discount factor value: gamma in [0.9, 0.95]
                pass
            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'))
            else:
                # java terminates unexpectedly
                pass

    print("finish training")