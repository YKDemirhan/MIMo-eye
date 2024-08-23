
import gymnasium as gym
import mimoEnv
from stable_baselines3 import PPO

env = gym.make("MIMoSelfBody-v0")

model = PPO("MultiInputPolicy", env,device="cuda", verbose=1, tensorboard_log="./tensorboard")
model.learn(1000000, log_interval=1)
model.save("ppo_agent_env")

#env.reset()
#for i in range (1000):
#    action = env.action_space.sample()
#    observation, reward, terminated, truncated, info = env.step(action)
#    print ("Observation: ", observation)
#    print ("Reward: ", reward)
#    print ("terminated", terminated)
#    print ("Truncated", truncated)
#    print ("Environment Info", info)
#    if (truncated == True):
#        print ("Episode Ended (Total Reward:", reward, ")")
#        break
#
#    env.render()
