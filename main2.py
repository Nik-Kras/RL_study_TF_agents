import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import wrappers
import Environment
import numpy as np

game = Environment.GridWorld(12)

# Get information about the environment
print(game.action_spec())
print(game.observation_spec())
print(game.map_spec())
print(game.time_step_spec_grid_world())

print("Map before RESET:")
print(game.state_matrix)

time_stamp = game.reset()

print("Map after RESET:")
print(game.state_matrix)


# Environment validation with 5 epochs
utils.validate_py_environment(game, episodes=5)

# Test playing the game
get_new_card_action = np.array(0, dtype=np.int32)
end_round_action = np.array(1, dtype=np.int32)

environment = game
time_step = environment.reset()
print(time_step.observation)
cumulative_reward = time_step.reward

for _ in range(3):
  time_step = environment.step(get_new_card_action)
  print(time_step.observation)
  print(game.position)
  cumulative_reward += time_step.reward

time_step = environment.step(end_round_action)
print(time_step.observation)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)

# Environment Wrappers
# It creates a modification of the environment
# There are many different wrappers available - environments/wrappers.py

# Terminate the game after a certain number of moves
wrapped_game = wrappers.TimeLimit(game, duration=50)

# TFPyEnvironment - wraps Python environment to TensorFlow env, which enables parallelisation
tf_env = tf_py_environment.TFPyEnvironment(wrapped_game)

print("***************************** TensorFlow Wrapper is applied ****************************")
print(isinstance(tf_env, tf_environment.TFEnvironment))
print("TimeStep Specs:", tf_env.time_step_spec())
print("Action Specs:", tf_env.action_spec())

time_step = tf_env.reset()
num_steps = 3
transitions = []
reward = 0
for i in range(num_steps):
  action = tf.constant([i % 2])
  # applies the action and returns the new TimeStep.
  next_time_step = tf_env.step(action)
  transitions.append([time_step, action, next_time_step])
  reward += next_time_step.reward
  time_step = next_time_step

np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
print('\n'.join(map(str, np_transitions)))
print('Total reward:', reward.numpy())


print("*************************** Let TensorFlow play a whole episode *******************")

tf_env = tf_py_environment.TFPyEnvironment(wrapped_game)

time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 5

for i in range(num_episodes):
  print("********************************************************")
  print("******* Eposide: " + str(i+1) + "/" + str(num_episodes))
  print("********************************************************")
  episode_reward = 0
  episode_steps = 0
  while not time_step.is_last():
    action = tf.random.uniform([1], 0, 4, dtype=tf.int32)
    time_step = tf_env.step(action)
    episode_steps += 1
    episode_reward += time_step.reward.numpy()
  print("Steps per episode: " + str(episode_steps))
  print("Reward per episode: " + str(episode_reward))
  rewards.append(episode_reward)
  steps.append(episode_steps)
  time_step = tf_env.reset()

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)