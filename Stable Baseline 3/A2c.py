from stable_baselines3 import A2C
import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=False)
env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 10

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        next_state, reward, done, _, _ = env.step(env.action_space.sample())
    print(reward)


env.close