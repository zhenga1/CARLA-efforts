import carla
import numpy as np
import random
import time
import torch
import torch.nn as nn
from torch.distributions import Normal

from env.carla_env import CarlaLaneEnv

def compute_gae(rewards, dones, values, last_values, gamma=0.99, lam=0.95):
    """
    Docstring for compute_gae

    :param rewards: list of rewards
    :param dones: list of done flags
    :param values: list of value estimates
    :param last_values: value estimate for the last state
    :param gamma: discount factor
    :param lam: GAE lambda
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    
    for t in reversed(range(T)):
        # last_values is some expression that is the value estimate for the state agent at the end
        # probably some sort of end episodic reward?
        next_value = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        # delta is the one step advantage
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        # some sort of moving average, so the advantage is accummulated 
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    
    final_returns = advantages + values
    return advantages, final_returns

def ppo_update(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    rollout,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
    batch_size=128
):
    """
    rollout: list of (obs, steer, throttle, logp, reward, done, value) tuples
    """
    device = next(model.parameters()).device # efficient way to get the model device
    # -- unpack the entire rollout --
    obs, steer, throttle, logp_old, rewards, dones, values = zip(*rollout)

    obs = torch.stack(obs).to(device) # (T, C, H, W), turn a list of tensors into a single tensor
    actions = torch.tensor(
        list(zip(steer, throttle)), 
        dtype=torch.float32,
        device=device
    ) # (T, 2)

    # -- convert to tensors via stacking --
    print(f"obs shape: {obs.shape}, actions shape: {actions.shape}")
    old_logp = torch.tensor(logp_old, dtype=torch.float32, device=device) # (T,)
    print("Old logp shape:", old_logp.shape)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device) # (T,)
    print("Rewards shape:", rewards.shape)
    dones = torch.tensor(dones, dtype=torch.float32, device=device) # (T,)
    print("Dones shape:", dones.shape)
    values = torch.tensor(values, dtype=torch.float32, device=device).squeeze(-1) # (T,)
    print(f"values shape: {values.shape}")

    # --- bootstrapping value ---
    with torch.no_grad():
        #print("shape of last obs:", obs[-1].unsqueeze(0).shape)
        _,_, last_value = model(obs[-1].unsqueeze(0).to(device=device))
        last_value = last_value.squeeze(-1)
    
    # compute the GAE advantages and returns
    advantages, returns = compute_gae(
        rewards, dones, values, last_value
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages

    T = obs.size(0)
    indices = torch.randperm(T) # Random permutations of integers

    # --- PPO optimization ---
    for _ in range(epochs):
        for start in range(0, T, batch_size):
            idx = indices[start:start+batch_size]

            # -- mini-batch, getting some of the rollout data --
            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_logp = old_logp[idx]
            mb_adv = advantages[idx]
            mb_returns = returns[idx]

            # -- forward pass --
            mu, std, value_pred = model(mb_obs)
            dist = Normal(mu, std)

            # -- inverse squashing (squash the steer + throttle to get a valid range of control) --
            steer = mb_actions[:,0].clamp(-0.999, 0.999)
            throttle = mb_actions[:,1].clamp(0.0001, 0.9999)

            raw0 = torch.atanh(steer)  # inverse tanh, bring the value back to real line
            raw1 = torch.log(throttle / (1 - throttle))  # inverse sigmoid
            raw_action = torch.stack([raw0, raw1], dim=-1) # get the original output 

            # Re-compute log prob
            new_logp = dist.log_prob(raw_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Old pro
            ratio = torch.exp(new_logp - mb_old_logp) # importance sampling ratio

            # --- PPO Objective ---
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- Value function loss ---
            value_loss = value_coef * (mb_returns - value_pred).pow(2).mean()
            entropy_loss = -entropy_coef * entropy.mean()

            # --- Total Accumulated Loss ---
            loss = policy_loss + value_loss + entropy_loss

            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()









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
        # we use a fixed std parameter (as 0). We start it from 0 (so std=1)
        # separate the log_std from the mu output so its learnable but separate from mu
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
    
    def get_action_and_value(self, obs_bchw):
        mu, std, value = self.forward(obs_bchw)
        dist = Normal(mu, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1) # sum on the action dimension

        # squash to get to the valid control range
        steer = torch.tanh(raw_action[:,0])  # ensure steer is in [-1,1]
        throttle = torch.sigmoid(raw_action[:,1])  # ensure throttle is in [0,1]

        action = torch.stack([steer, throttle], dim=-1)
        return action, log_prob, entropy, value

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
    