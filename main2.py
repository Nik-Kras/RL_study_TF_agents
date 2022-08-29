import Environment
from tf_agents.environments import utils
from MapGenerator.Grid import *
import numpy as np

game = Environment.GridWorld(30, 30)

# Get information about the environment
print(game.action_spec())
print(game.observation_spec())
print(game.map_spec())
print(game.time_step_spec_grid_world())

# Generate the Map
Generator = Grid(10)    # How many 3x3 tiles should be put in the Map
state_matrix = Generator.GenerateMap() - 1
game.setStateMatrix(state_matrix)

# Environment validation with 5 epochs
utils.validate_py_environment(game, episodes=5)

# Test playing the game
get_new_card_action = np.array(0, dtype=np.int32)
end_round_action = np.array(1, dtype=np.int32)

environment = Environment.GridWorld(30, 30)
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(3):
  time_step = environment.step(get_new_card_action)
  print(time_step)
  cumulative_reward += time_step.reward

time_step = environment.step(end_round_action)
print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)

