import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy

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

print("*************************** Policy development ************************")

print("Random Policy test")
action_spec = game.action_spec()
my_random_py_policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
    action_spec=action_spec)
time_step = None
action_step = my_random_py_policy.action(time_step)
print(action_step)
action_step = my_random_py_policy.action(time_step)
print(action_step)
action_step = my_random_py_policy.action(time_step)
print(action_step)

print("Scripted Policy")

action_spec = game.action_spec()
action_script = [(1, np.array(2, dtype=np.int32)),
                 (0, np.array(0, dtype=np.int32)), # Setting `num_repeats` to 0 will skip this action.
                 (2, np.array(1, dtype=np.int32)),
                 (1, np.array(3, dtype=np.int32))]

my_scripted_py_policy = scripted_py_policy.ScriptedPyPolicy(
    time_step_spec=None, action_spec=action_spec, action_script=action_script)

policy_state = my_scripted_py_policy.get_initial_state()
time_step = None
print('Executing scripted policy...')
action_step = my_scripted_py_policy.action(time_step, policy_state)
print(action_step)
action_step= my_scripted_py_policy.action(time_step, action_step.state)
print(action_step)
action_step = my_scripted_py_policy.action(time_step, action_step.state)
print(action_step)

print('Resetting my_scripted_py_policy...')
policy_state = my_scripted_py_policy.get_initial_state()
action_step = my_scripted_py_policy.action(time_step, policy_state)
print(action_step)

print("************* TF policies ***********")
action_spec = tensor_spec.BoundedTensorSpec(
    (2,), tf.float32, minimum=-1, maximum=3)
input_tensor_spec = tensor_spec.BoundedTensorSpec((2,), tf.int32, minimum=-1, maximum=1)
time_step_spec = ts.time_step_spec(input_tensor_spec)

print("action_spec")
print(action_spec)
print("input_tensor_spec")
print(input_tensor_spec)
print("time_step_spec")
print(time_step_spec)

my_random_tf_policy = random_tf_policy.RandomTFPolicy(
    action_spec=action_spec, time_step_spec=time_step_spec)
observation = tf.ones(time_step_spec.observation.shape, dtype=time_step_spec.observation.dtype)
time_step = ts.restart(observation)
action_step = my_random_tf_policy.action(time_step)

print('Observation:')
print(observation)
print('Action:')
print(action_step.action)

print("******************** DQN ********************")

input_tensor_spec = tensor_spec.TensorSpec((4,), tf.float32)
time_step_spec = ts.time_step_spec(input_tensor_spec)
action_spec = tensor_spec.BoundedTensorSpec((),
                                            tf.int32,
                                            minimum=0,
                                            maximum=3)
num_actions = action_spec.maximum - action_spec.minimum + 1

print(time_step_spec)
print(time_step_spec.observation)
print(time_step_spec.observation.shape.as_list())

print("_________________________________")

batch_size = 3
shape1 = batch_size
shapeList = time_step_spec.observation.shape.as_list()
shape2 = shapeList[0]
# shape3 = shapeList[1]

print("Shape: " + str([shape1] + shapeList))

observation = tf.ones(shape=([shape1] + shapeList))
time_steps = ts.restart(observation, batch_size=batch_size)

print("observation")
print(observation)
print("time_steps")
print(time_steps)

class QNetwork(network.Network):

  def __init__(self, input_tensor_spec, action_spec, num_actions, name=None):
    super(QNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
    self._sub_layers = [
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(num_actions),
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._sub_layers:
      inputs = layer(inputs)
    return inputs, network_state


my_q_network = QNetwork(
    input_tensor_spec=input_tensor_spec,
    action_spec=action_spec,
    num_actions=num_actions)

print("----- Check Q-Network -------")
print(my_q_network)
my_q_network.build(input_shape=(12,12))
print(my_q_network.summary())
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

my_q_policy = q_policy.QPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    q_network=my_q_network)


action_step = my_q_policy.action(time_steps)
distribution_step = my_q_policy.distribution(time_steps)

print('Action:')
print(action_step.action)

print('Action distribution:')
print(distribution_step.action)