import carla
import numpy as np
import random
import time
import torch
import torch.nn as nn
from torch.distributions import Normal

from env.carla_env import CarlaLaneEnv

class ActorCritic (torch.nn.Module):
    def __init__(self, obs_shape, feature_dim=512):
        super(ActorCritic, self).__init__()
        
        # define feature dim
        self.feature_dim = feature_dim

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)  # (1, C, H, W)
            conv_out = self.encoder(dummy_input)
            # use dummy input to infer the size of the conv output
            conv_out_size = conv_out.view(1, -1).shape[1]
        
        # pass the flattened conv output through a fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, feature_dim),
            nn.ReLU()
        )

        # Actor head
        self.mu = nn.Linear(feature_dim, 2) # steering and throttle output
        self.log_std = nn.Parameter(torch.zeros(2)) # log std for action distribution

        # Critic head
        self.value = nn.Linear(feature_dim, 1)


    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1) # flatten everything except the first dimension, which is batch dim
        x = self.fc(x)

        # Actor model 
        mu = self.mu(x)
        std = self.log_std.exp()

        # Critic model
        value = self.value(x)
        return mu, std, value

class CarlaLaneEnvPPO (CarlaLaneEnv):
    def __init__(self, max_steps=500):
        super().__init__(max_steps)
    
    def _get_obs(self):
        # An internal function that returns a tensorized version of the current image
        img = self.image # (H, W, C) numpy array (assumed to be uint8)
        if img is None:
            return None
        img = img.astype(np.float32) / 255.0  # normalize to [0,1]
        img = torch.from_numpy(img).permute(2,0,1) # (C, H, W)
        self.image_shape = img.shape
        return img

    def reset(self):
        obs = super().reset()
        return self._get_obs()

    
    def step(self, steer, throttle, first_person=False):
        obs, reward, done = super().step(steer, throttle, first_person)
        return self._get_obs(), reward, done

    @staticmethod
    def act(model, obs):
        # assume observation is just a single image tensor (C, H, W)
        mu, std, value = model(obs.unsqueeze(0)) # add batch dim
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        steer = torch.tanh(action[0,0])  # ensure steer is in [-1,1]
        throttle = torch.sigmoid(action[0,1])  # ensure throttle is in [0,1]

        return steer.item(), throttle.item(), log_prob.item(), value.item()
    