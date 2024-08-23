import gymnasium as gym
from stable_baselines3 import SAC
import mimoEnv
import numpy as np

def test_model(env, model_path, test_episodes=10, max_steps=1000, deterministic=True):
    model = SAC.load(model_path)
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

        print(f"Episode {episode + 1}/{test_episodes} - Total Reward: {total_reward}, Steps: {steps}")

    success_rate = success_count / test_episodes
    average_reward = np.mean(total_rewards)
    average_steps = np.mean(steps_to_success) if steps_to_success else float('inf')

    print("\n=== Evaluation Results ===")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Total Reward: {average_reward:.2f}")
    print(f"Average Steps to Success: {average_steps:.2f}")

if __name__ == "__main__":
    env = gym.make("MIMoSelfBody-v0", render_mode="human")
    test_model(env, "sac_agent_env3", test_episodes=10, max_steps=1000, deterministic=True)







#import gymnasium as gym
#from stable_baselines3 import SAC
#import mimoEnv

#env = gym.make("MIMoSelfBody-v0", render_mode="human")

#model = SAC("MultiInputPolicy", env, verbose=1)

#model.load("sac_agent_env")

#obs, info = env.reset()
#while True:
#    action, _states = model.predict(obs, deterministic=True)
#    obs, reward, terminated, truncated, info = env.step(action)
#    print ("Reward: ", reward)
#    env.render()
#    if (terminated or truncated):
#        break

#tensorboard --logdir ./tensorboard