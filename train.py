import numpy as np
from env.carla_env import CarlaLaneEnv
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
    argparse.add_argument(
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

    env = CarlaLaneEnv()
    obs = env.reset()
    # No training at the moment
    # Just trying to test environment interaction
    for episode in range(num_episodes):
        print(f"=== Episode {episode} ===")
        
        obs = env.reset() # get the current image
        done = False
        episode_reward = 0.0  # accumulated reward for this specific episode

        for step in range(steps):
            steer = np.random.uniform(-0.2, 0.2)
            throttle = 0.5

            # obs is an image, reward is speed, done is always False
            obs, reward, done = env.step(steer, throttle, first_person=first_person)
            print(f"Step {step} | reward={reward:.2f}")

            time.sleep(0.05)
        
    env.close()