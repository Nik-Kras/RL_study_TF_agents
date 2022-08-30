import Environment
from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import wrappers
from MapGenerator.Grid import *
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