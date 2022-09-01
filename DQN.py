import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.policies import q_policy

import numpy as np

import Environment

game = Environment.GridWorld(12)

# Get information about the environment
print(game.action_spec())
print(game.map_spec())

time_stamp = game.reset()
wrapped_game = wrappers.TimeLimit(game, duration=50)
tf_env = tf_py_environment.TFPyEnvironment(wrapped_game)

print("***************************** TensorFlow Wrapper is applied ****************************")
print(isinstance(tf_env, tf_environment.TFEnvironment))
print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())

input_tensor_spec = tensor_spec.TensorSpec(shape=[1, 12, 12], dtype=tf.int32)
time_step_spec = ts.time_step_spec(input_tensor_spec)

print("New TimeStep Spec:", time_step_spec)

# time_step_spec = tf_env.time_step_spec()
# time_step_spec = tf.expand_dims(time_step_spec, axis=-1)        # To include place for batch (12,12) -> (12,12,)
action_spec = tensor_spec.from_spec(tf_env.action_spec())
num_actions = action_spec.maximum - action_spec.minimum + 1

print("_________________________________")
print("time_step_spec.observation:")
print(time_step_spec.observation)
print(time_step_spec.observation.shape.as_list())
print("_________________________________")

# batch_size = 2
# shape1 = batch_size
# shapeList = time_step_spec.observation.shape.as_list()
# shape2 = shapeList[0]
# shape3 = shapeList[1]

# print("Shape: " + str([batch_size, 12, 12]))


# shape_of_element = tf_env.time_step_spec().observation.shape
# observation = tf.ones(shape=[batch_size, 12, 12])
time_steps = tf_env.reset()  #ts.restart(observation, batch_size=batch_size)

print("observation: ", time_steps.observation)
print("time_steps: ", time_steps)

class QNetwork(network.Network):

  def __init__(self, input_tensor_spec, action_spec, num_actions, name=None):
    super(QNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
    self._sub_layers = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_actions)
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    print("called input: ", inputs)
    my_observation = inputs
    print("my_observation: ", my_observation)
    inputs = tf.cast(my_observation, tf.float32)
    print("Final input: ", inputs)
    for layer in self._sub_layers:
      inputs = layer(inputs)
    print("Final output: ", inputs)
    return inputs, network_state

print("---- time_step_spec.observation", time_step_spec.observation)

my_q_network = QNetwork(
    input_tensor_spec=tf_env.observation_spec(),
    action_spec=action_spec,
    num_actions=num_actions)

# print("----- Check Q-Network -------")
# print(my_q_network)
# my_q_network.__call__(observation)  # Important! To initialize weights and to set dimentions (shapes) in NN
# print(my_q_network.summary())
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

my_q_policy = q_policy.QPolicy(
    time_step_spec=tf_env.time_step_spec(),
    action_spec=action_spec,
    q_network=my_q_network)


action_step = my_q_policy.action(time_steps)
distribution_step = my_q_policy.distribution(time_steps)

print('Action:')
print(action_step.action)

print('Action distribution:')
print(distribution_step.action)