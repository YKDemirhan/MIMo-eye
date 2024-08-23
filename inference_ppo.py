

import gymnasium as gym
from stable_baselines3 import PPO
import mimoEnv
import numpy as np

def test_model(env, model, num_episodes=100, render=False):
    successes = 0
    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            if render:
                env.render()
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        if terminated:
            successes += 1

    success_rate = successes / num_episodes
    average_total_reward = np.mean(total_rewards)
    average_steps = np.mean(total_steps)
    
    return success_rate, average_total_reward, average_steps

def main():
    env = gym.make("MIMoSelfBody-v0", render_mode="human")
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Load the trained model
    model = PPO.load("ppo_agent")

    # Test the model
    success_rate, average_total_reward, average_steps = test_model(env, model, num_episodes=100, render=True)
    
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Total Reward: {average_total_reward:.2f}")
    print(f"Average Steps to Success: {average_steps:.2f}")

if __name__ == "__main__":
    main()

