import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym
from collections import deque
from fightingice_env import FightingiceEnv
import os


# * an implement for rainbow dqn with double, dueling, noisy, c51(categorical/distribution), multi-step.
# * without prioritized replay buffer
class n_step_replay_buffer(object):
    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=self.capacity)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def get_n_step_info(self):
        observation, action, reward, next_observation, done = self.n_step_buffer[
            -1]

        for _, _, rew, next_obs, do in reversed(list(self.n_step_buffer)[:-1]):
            reward = reward * self.gamma * (1 - do) + rew
            next_observation, done = (next_obs,
                                      do) if do else (next_observation, done)

        return reward, next_observation, done

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.n_step_buffer.append(
            [observation, action, reward, next_observation, done])
        if len(self.n_step_buffer) < self.n_step:
            return

        observation, action = self.n_step_buffer[0][:2]
        reward, next_observation, done = self.get_n_step_info()
        self.memory.append(
            [observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(*batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(
            next_observation, 0), done

    def __len__(self):
        return len(self.memory)

    # * change for fighting game
    def change_done(self):
        observation, action, reward, next_observation, done = self.n_step_buffer[
            -1]
        observation = list(observation[0])
        next_observation = list(next_observation[0])
        done = True
        self.n_step_buffer.pop()
        self.memory.pop()
        self.store(observation, action, reward, next_observation, done)

    def clear(self):
        self.n_step_buffer.clear()


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


def projection_distribution(target_model, next_observation, reward, done,
                            v_min, v_max, atoms_num, gamma):
    batch_size = next_observation.size(0)
    delta_z = float(v_max - v_min) / (atoms_num - 1)
    support = torch.linspace(v_min, v_max, atoms_num)

    next_dist = target_model.forward(next_observation) * support
    next_dist = next_dist.detach()
    next_action = next_dist.sum(2).max(1)[1].detach()
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(
        batch_size, 1, atoms_num)
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(next_dist)
    done = done.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = reward + (1 - done) * gamma * support
    Tz = Tz.clamp(min=v_min, max=v_max)
    b = (Tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(
        0, (batch_size - 1) * atoms_num,
        batch_size).long().unsqueeze(1).expand_as(next_dist)

    proj_dist = torch.zeros_like(next_dist, dtype=torch.float32)
    proj_dist.view(-1).index_add_(0, (offset + l).view(-1),
                                  (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (offset + l).view(-1),
                                  (next_dist * (b - l.float())).view(-1))
    return proj_dist


def train(eval_model, target_model, buffer, v_min, v_max, atoms_num, gamma,
          batch_size, optimizer, count, update_freq):
    observation, action, reward, next_observation, done = buffer.sample(
        batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    proj_dist = projection_distribution(target_model, next_observation, reward,
                                        done, v_min, v_max, atoms_num, gamma)

    dist = eval_model.forward(observation)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, atoms_num)
    dist = dist.gather(1, action).squeeze(1)
    dist.detach().clamp_(0.01, 0.99)
    loss = -(proj_dist * dist.log()).sum(1).mean()
    # * get the grad of dist

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    eval_model.reset_noise()
    target_model.reset_noise()

    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    episode = 1000
    epsilon_init = 0.30
    epsilon_decay = 0.996
    epsilon_min = 0.01
    update_freq = 100
    gamma = 0.92
    learning_rate = 1e-3
    atoms_num = 51
    v_min = -10
    v_max = 10
    batch_size = 64
    capacity = 50000
    exploration = 200
    n_step = 2

    env = FightingiceEnv(port=4240)
    env_args = [
        "--fastmode", "--disable-window", "--grey-bg", "--inverted-player",
        "1", "--mute"
    ]
    action_dim = env.action_space.n
    observation_dim = env.observation_space.shape[0]
    count = 0
    if not os.path.exists('./log_data/rainbow.npz'):
        target_net = rainbow_dqn(observation_dim, action_dim, atoms_num, v_min,
                                 v_max)
        eval_net = rainbow_dqn(observation_dim, action_dim, atoms_num, v_min,
                               v_max)
        target_net.load_state_dict(eval_net.state_dict())
        reward_list = []
        weight_reward_list = []
        epsilon = epsilon_init
        weight_reward = None
        i = 0
    else:
        target_net = torch.load('model/rainbow.pkl')
        eval_net = torch.load('model/rainbow.pkl')
        log_data = np.load('./log_data/rainbow.npz')
        epsilon = log_data['epsilon']
        reward_list = list(log_data['reward_list'])
        weight_reward_list = list(log_data['weight_reward_list'])
        weight_reward = weight_reward_list[-1]
        i = len(reward_list)
        print('load data finish!!!')
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = n_step_replay_buffer(capacity, n_step, gamma)

    win_count = 0
    done_flag = True
    while True:
        if not done_flag:
            buffer.clear()
        done_flag = False
        obs = env.reset(env_args=env_args)
        reward_total = 0
        i = i + 1
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)),
                                  epsilon)
            next_obs, reward, done, info = env.step(action)
            count += 1
            if next_obs is not None and obs is not None:
                buffer.store(obs, action, reward / 10., next_obs, done)
            elif info is None:
                i = i - 1
                buffer.clear()
                break
            reward_total += reward
            obs = next_obs
            if len(buffer) > exploration:
                train(eval_net, target_net, buffer, v_min, v_max, atoms_num,
                      gamma, batch_size, optimizer, count, update_freq)
            if done:
                done_flag = True
                buffer.change_done()
                if epsilon > epsilon_min:
                    epsilon = epsilon * epsilon_decay
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.9 * weight_reward + 0.1 * reward_total
                if info[0] > info[1]:
                    win_count += 1
                reward_list.append(reward_total)
                weight_reward_list.append(weight_reward)
                torch.save(eval_net, './model/rainbow.pkl')
                np.savez('./log_data/rainbow.npz',
                         reward_list=reward_list,
                         weight_reward_list=weight_reward_list,
                         epsilon=epsilon)
                print(
                    'episode: {}  reward: {}  weight_reward: {:.3f}  epsilon: {:.2f}  win_count: {}'
                    .format(i + 1, reward_total, weight_reward, epsilon,
                            win_count))
                break

