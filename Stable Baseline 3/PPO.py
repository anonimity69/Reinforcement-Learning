from stable_baselines3 import PPO
import gymnasium as gym
import os

env = gym.make("CarRacing-v3", render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=False)
env.reset()

model_dir = 'models/PPO'
logdir = 'log'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{model_dir}/{TIMESTEPS*i}")

# episodes = 10

# for episode in range(episodes):
#     state, _ = env.reset()
#     done = False
#     while not done:
#         env.render()
#         next_state, reward, done, _, _ = env.step(env.action_space.sample())
#     print(reward)



env.close