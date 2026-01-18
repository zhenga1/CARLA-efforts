import numpy as np
from env.carla_env_ppo import CarlaLaneEnvPPO, ActorCritic
import time
import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    argparser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='Number of steps to run EACH episode in the environment'
    )
    argparser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of episodes to run'
    )
    argparser.add_argument(
        '--firstperson',
        action='store_true',
        help='First person view from the vehicle'
    )

    args = argparser.parse_args()
    
    # deal with parameters
    seed = args.seed
    np.random.seed(seed)

    steps = args.steps
    first_person = args.firstperson

    num_episodes = args.episodes

    rollout = []

    env = CarlaLaneEnvPPO()
    obs = env.reset()

    model = ActorCritic(obs.shape)
    
    # No training at the moment
    # Just trying to test environment interaction
    for episode in range(num_episodes):
        print(f"=== Episode {episode} ===")
        obs = env.reset() # get the current image
        done = False
        episode_reward = 0.0  # accumulated reward for this specific episode

        while not done:
            steer = np.random.uniform(-0.2, 0.2)
            throttle = 0.5

            # choose an action for the model to take
            steer, throttle, logp, value = CarlaLaneEnvPPO.act(model, obs)

            # obs is an image, reward is speed, done is whether the episode has finished
            next_obs, reward, done = env.step(steer, throttle, first_person=first_person)

            # Rollout storage
            rollout.append((obs, steer, throttle, logp, reward, done, value))
            
            # move to the next observation
            obs = next_obs

            episode_reward += reward

            time.sleep(0.05) # visualization only
        print(f"Episode {episode} | reward={episode_reward:.2f}")
        
    env.close()