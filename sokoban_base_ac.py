import sys
sys.path.append("./common")

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from common.multiprocessing_env import SubprocVecEnv
from common.minipacman import MiniPacman
from common.environment_model import EnvModel
from common.actor_critic import OnPolicy, ActorCritic, RolloutStorage

import matplotlib.pyplot as plt

import gym
import gym_sokoban

import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_steps', type=int, default=8,
                    help='num of steps')
parser.add_argument('--num_envs', type=int, default=8,
                    help='num of cpus')
parser.add_argument('--id', type=str, default="default",
                    help='id')

args = parser.parse_args()
writer = SummaryWriter(f'results/{args.id}')

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

load_env_filename = None#"sokoban_i2a_env_sokoban_100_-19.87504005432129"
load_distill_filename = None#"sokoban_i2a_distill_sokoban_100_-19.87504005432129"
load_ac_filename = None#"sokoban_i2a_ac_sokoban_100_-19.87504005432129"

pixels = (
    (0, 0, 0),
    (243, 248, 238), 
    (254, 126, 125),
    (254, 95, 56),
    (142, 121, 56),
    (160, 212, 56), 
    (219, 212, 56)
)
pixel_to_onehot = {pix:i for i, pix in enumerate(pixels)} 
num_pixels = len(pixels)

task_rewards = {
    "regular": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "avoid":   [0.1, -0.1, -5, -10, -20],
    "hunt":    [0, 1, 10, -20],
    "ambush":  [0, -0.1, 10, -20],
    "rush":    [0, -0.1, 9.9],
    "sokoban": [-0.1, 0.9, -1.1, 9.9]
}
reward_to_onehot = {mode: {reward:i for i, reward in enumerate(task_rewards[mode])} for mode in task_rewards.keys()}

def pix_to_target(next_states):
    target = []
    for pixel in next_states.transpose(0, 2, 3, 1).reshape(-1, 3):
        target.append(pixel_to_onehot[tuple([np.round(pixel[0]), np.round(pixel[1]), np.round(pixel[2])])])
    return target

def target_to_pix(imagined_states):
    pixels = []
    to_pixel = {value: key for key, value in pixel_to_onehot.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))
    return np.array(pixels)

def rewards_to_target(mode, rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_onehot[mode][reward])
    return target

def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('loss %s' % losses[-1])
    plt.plot(losses)
    plt.show()
    
def displayImage(image, step, reward):
    s = str(step) + " " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()

mode = "sokoban"
num_envs = args.num_envs

class ChannelFirstEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (3, 10, 10))
    
    def observation(self, obs):
        obs = obs.transpose(2, 0, 1)
        return obs

def make_env():
    def _thunk():
        # env = MiniPacman(mode, 1000)
        env = ChannelFirstEnv(gym.make('Boxoban-Train-v0'))
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_shape = envs.observation_space.shape
num_actions = envs.action_space.n
num_rewards = len(task_rewards[mode])

#a2c hyperparams:
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
num_steps = 5
num_frames = int(10e5)

#rmsprop hyperparams:
lr    = 7e-4
eps   = 1e-5
alpha = 0.99

#Init a2c and rmsprop
actor_critic = ActorCritic(envs.observation_space.shape, envs.action_space.n)
optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
    
if USE_CUDA:
    actor_critic = actor_critic.cuda()

rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
rollout.cuda()

all_rewards = []
all_losses  = []

state = envs.reset()
state = torch.FloatTensor(np.float32(state))

rollout.states[0].copy_(state)

episode_rewards = torch.zeros(num_envs, 1)
final_rewards   = torch.zeros(num_envs, 1)

for i_update in range(num_frames):

    for step in range(num_steps):
        action = actor_critic.act(Variable(state))

        next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())

        reward = torch.FloatTensor(reward).unsqueeze(1)
        episode_rewards += reward
        masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1)
        final_rewards *= masks
        final_rewards += (1-masks) * episode_rewards
        episode_rewards *= masks

        if USE_CUDA:
            masks = masks.cuda()

        state = torch.FloatTensor(np.float32(next_state))
        rollout.insert(step, state, action.data, reward, masks)


    _, next_value = actor_critic(Variable(rollout.states[-1], volatile=True))
    next_value = next_value.data

    returns = rollout.compute_returns(next_value, gamma)

    logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
        Variable(rollout.states[:-1]).view(-1, *state_shape),
        Variable(rollout.actions).view(-1, 1)
    )

    values = values.view(num_steps, num_envs, 1)
    action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
    advantages = Variable(returns) - values

    value_loss = advantages.pow(2).mean()
    action_loss = -(Variable(advantages.data) * action_log_probs).mean()

    optimizer.zero_grad()
    loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
    loss.backward()
    nn.utils.clip_grad_norm(actor_critic.parameters(), max_grad_norm)
    optimizer.step()
    
    if i_update % 100 == 0:
        all_rewards.append(final_rewards.mean())
        all_losses.append(loss.item())
        
        print('epoch %s. reward: %s, loss: %s' % (i_update, all_rewards[-1].numpy(), all_losses[-1]))
        # plt.figure(figsize=(20,5))
        # plt.subplot(131)
        # plt.title('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))
        # plt.plot(all_rewards)
        # plt.subplot(132)
        # plt.title('loss %s' % all_losses[-1])
        # plt.plot(all_losses)
        # plt.show()
    
    if i_update % 100000 == 0:
        torch.save(actor_critic.state_dict(), "actor_critic_" + mode + str(i_update))        
        
    rollout.after_update()

torch.save(actor_critic.state_dict(), "actor_critic_" + mode)

env = ChannelFirstEnv(gym.make('Boxoban-Train-v0'))

done = False
state = env.reset()
total_reward = 0
step   = 1


while not done:
    current_state = torch.FloatTensor(state).unsqueeze(0)
    if USE_CUDA:
        current_state = current_state.cuda()
        
    action = actor_critic.act(Variable(current_state))
    
    next_state, reward, done, _ = env.step(action.data[0, 0])
    total_reward += reward
    state = next_state
    
    image = torch.FloatTensor(state).permute(1, 2, 0).cpu().numpy()
    displayImage(image, step, total_reward)
    step += 1