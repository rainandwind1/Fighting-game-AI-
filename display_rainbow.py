import random
from fightingice_env import FightingiceEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
import os


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.01):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(
            torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer(
            'weight_epsilon', torch.FloatTensor(self.output_dim,
                                                self.input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigam = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.register_buffer('bias_epsilon',
                             torch.FloatTensor(self.output_dim))

        self.reset_parameter()
        self.reset_noise()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(
                self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigam.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def reset_parameter(self):
        mu_range = 1. / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init /
                                         np.sqrt(self.input_dim))
        self.bias_sigam.detach().fill_(self.std_init /
                                       np.sqrt(self.output_dim))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))


class rainbow_dqn(nn.Module):
    def __init__(self, observation_dim, action_dim, atoms_num, v_min, v_max):
        super(rainbow_dqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.atoms_num = atoms_num
        self.v_min = v_min
        self.v_max = v_max

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 256)

        self.value_noisy1 = NoisyLinear(256, 256)
        self.value_noisy2 = NoisyLinear(256, self.atoms_num)

        self.adv_noisy1 = NoisyLinear(256, 256)
        self.adv_noisy2 = NoisyLinear(256, self.action_dim * self.atoms_num)

    def forward(self, observation):
        batch_size = observation.size(0)
        feature = F.relu(self.fc2(F.relu(self.fc1(observation))))

        value = self.value_noisy2(F.relu(self.value_noisy1(feature)))
        advantage = self.adv_noisy2(F.relu(self.adv_noisy1(feature)))

        value = value.view(batch_size, 1, self.atoms_num)
        advantage = advantage.view(batch_size, self.action_dim, self.atoms_num)

        dist = value + advantage - advantage.mean(1, keepdim=True)
        dist = F.softmax(dist, 2)
        return dist

    def reset_noise(self):
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            dist = self.forward(observation).detach()
            dist = dist * torch.linspace(self.v_min, self.v_max,
                                         self.atoms_num)
            action = dist.sum(2).max(1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action


if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    #env_args = ["--fastmode", "--grey-bg", "--inverted-player", "3", "--mute"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]
    model = torch.load('model/rainbow.pkl')
    model.training = False
    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None

        while not done:
            act = model.act(torch.FloatTensor(np.expand_dims(obs, 0)), 0)
            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            new_obs, reward, done, info = env.step(act)

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
