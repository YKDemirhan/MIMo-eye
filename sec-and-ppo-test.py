import gymnasium as gym
from stable_baselines3 import SAC, PPO
import mimoEnv  # Ensure this is imported correctly
import numpy as np

def test_model(env, model, test_episodes=100, max_steps=1000, deterministic=True):
    success_count = 0
    total_rewards = []
    steps_to_success = []

    for episode in range(test_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        while steps < max_steps:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        if terminated:
            success_count += 1
            steps_to_success.append(steps)

    success_rate = success_count / test_episodes
    average_reward = np.mean(total_rewards)
    average_steps = np.mean(steps_to_success) if steps_to_success else float('inf')

    return success_rate, average_reward, average_steps

if __name__ == "__main__":
    env = gym.make("MIMoSelfBody-v0", render_mode="human")

    # Test the SAC model
    print("Testing SAC Model")
    sac_model = SAC.load("sac_agent_env3")
    sac_success_rate, sac_average_reward, sac_average_steps = test_model(env, sac_model, test_episodes=100, max_steps=1000, deterministic=True)

    # Test the PPO model
    print("\nTesting PPO Model")
    ppo_model = PPO.load("ppo_agent")
    ppo_success_rate, ppo_average_reward, ppo_average_steps = test_model(env, ppo_model, test_episodes=100, max_steps=1000, deterministic=True)

    # Print final results
    print("\n=== Final Evaluation Results ===")
    print("SAC Model Results:")
    print(f"Success Rate: {sac_success_rate * 100:.2f}%")
    print(f"Average Total Reward: {sac_average_reward:.2f}")
    print(f"Average Steps to Success: {sac_average_steps:.2f}")

    print("\nPPO Model Results:")
    print(f"Success Rate: {ppo_success_rate * 100:.2f}%")
    print(f"Average Total Reward: {ppo_average_reward:.2f}")
    print(f"Average Steps to Success: {ppo_average_steps:.2f}")
