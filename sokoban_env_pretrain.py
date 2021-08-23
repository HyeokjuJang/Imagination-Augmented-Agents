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
from common.environment_model import EnvModelSokoban as EnvModel
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
parser.add_argument('--test', action='store_true',
                    help='if only test')

args = parser.parse_args()
writer = SummaryWriter(f'results/{args.id}')

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

load_env_filename = None
load_ac_filename = "actor_critic_sokoban"

if args.test:
    load_env_filename = "env_model_sokoban_env_fixed"
    load_ac_filename = "actor_critic_sokoban"

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
    "sokoban": [-0.1, 0.9, -1.1, 9.9, 10.9]
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

env_model     = EnvModel(envs.observation_space.shape, num_pixels, num_rewards)
actor_critic = ActorCritic(envs.observation_space.shape, envs.action_space.n)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(env_model.parameters())

if USE_CUDA:
    env_model     = env_model.cuda()
    actor_critic  = actor_critic.cuda()

actor_critic.load_state_dict(torch.load(load_ac_filename))
if args.test:
    env_model.load_state_dict(torch.load(load_env_filename))

def get_action(state):
    if state.ndim == 4:
        state = torch.FloatTensor(np.float32(state))
    else:
        state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
        
    action = actor_critic.act(Variable(state, volatile=True))
    action = action.data.cpu().squeeze(1).numpy()
    return action

def play_games(envs, frames):
    states = envs.reset()
    
    for frame_idx in range(frames):
        actions = get_action(states)
        next_states, rewards, dones, _ = envs.step(actions)
        
        yield frame_idx, states, actions, rewards, next_states, dones
        
        states = next_states

reward_coef = 0.1
num_updates = 1000000

losses = []
all_rewards = []

if not args.test:
    for frame_idx, states, actions, rewards, next_states, dones in play_games(envs, num_updates):
        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions)

        batch_size = states.size(0)
        
        onehot_actions = torch.zeros(batch_size, num_actions, *state_shape[1:])
        onehot_actions[range(batch_size), actions] = 1
        inputs = Variable(torch.cat([states, onehot_actions], 1))
        
        if USE_CUDA:
            inputs = inputs.cuda()
        imagined_state, imagined_reward = env_model(inputs)

        target_state = pix_to_target(next_states)
        target_state = Variable(torch.LongTensor(target_state))

        target_reward = rewards_to_target(mode, rewards)
        target_reward = Variable(torch.LongTensor(target_reward))

        optimizer.zero_grad()
        image_loss  = criterion(imagined_state, target_state)
        reward_loss = criterion(imagined_reward, target_reward)
        loss = image_loss + reward_coef * reward_loss
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        all_rewards.append(np.mean(rewards))
        
        if frame_idx % 100 == 0:
            print('epoch %s. reward: %s, loss: %s' % (frame_idx, all_rewards[-1], losses[-1]))
            #plot(frame_idx, all_rewards, losses)

    torch.save(env_model.state_dict(), "env_model_" + mode)

import time

env = ChannelFirstEnv(gym.make('Boxoban-Train-v0'))
batch_size = 1

done = False
state = env.reset()
iss = []
ss  = []

steps = 0

while not done:
    steps += 1
    actions = get_action(state)
    onehot_actions = torch.zeros(batch_size, num_actions, *state_shape[1:])
    onehot_actions[range(batch_size), actions] = 1
    state = torch.FloatTensor(state).unsqueeze(0)
    
    inputs = Variable(torch.cat([state, onehot_actions], 1))
    if USE_CUDA:
        inputs = inputs.cuda()

    imagined_state, imagined_reward = env_model(inputs)
    imagined_state = F.softmax(imagined_state)
    iss.append(imagined_state)
    
    next_state, reward, done, _ = env.step(actions[0])
    ss.append(state)
    state = next_state
    
    imagined_image = target_to_pix(imagined_state.view(batch_size, -1, len(pixels))[0].max(1)[1].data.cpu().numpy())
    imagined_image = imagined_image.reshape(10, 10, 3)
    state_image = torch.FloatTensor(next_state).permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.title("Imagined")
    plt.imshow(imagined_image)
    plt.subplot(132)
    plt.title("Actual")
    plt.imshow(np.array(state_image, dtype=np.uint8))
    plt.show()
    time.sleep(0.3)
    
    if steps > 200:
        break