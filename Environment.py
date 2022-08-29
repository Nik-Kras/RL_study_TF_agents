import numpy as np
import abc
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

"""
To use that environment do next:

1. Create an object based on the class and set the desired parameters of the game 
>>> import Environment
>>> game = Environment.GridWorld(tot_row = 30, tot_col = 30)
2. Create your own map of walls. 
- It must be a matrix of the same size as a game (30x30)
- The values of matrix have next meaning: 
    0  walkable path, 
    -1 wall
- Don't put anything else
- You could use Map Generator provided separately
>>> from MapGenerator.Grid import *
>>> Generator = Grid(SIZE)
>>> state_matrix = Generator.GenerateMap() - 1
3. Set the map according to your desired walls configuration
>>> game.setStateMatrix(state_matrix)
4. Set player and goals position randomly
>>> game.setPosition()
5. To view the world use
>>> game.render()
6. To read the world use
>>> game.getWorldState()
7. To make an action by agent use
>>> game.step(action) 
- That will return you observation of the world, 
- Therefore, at that moment you don't need any other functions besides step() and render()
8. To create a new game clear the environment
>>> game.clear()
9. Then, repeat from step #2
"""

# Updated GridWorld to standards
# To be wrapped by TensorFlow environment
class GridWorld(py_environment.PyEnvironment):

    # def __init__(self):
    #     self._action_spec = array_spec.BoundedArraySpec(
    #         shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    #     self._observation_spec = array_spec.BoundedArraySpec(
    #         shape=(1,), dtype=np.int32, minimum=0, name='observation')
    #     self._state = 0
    #     self._episode_ended = False
    #
    # def action_spec(self):
    #     return self._action_spec
    #
    # def observation_spec(self):
    #     return self._observation_spec
    #
    # def _reset(self):
    #     self._state = 0
    #     self._episode_ended = False
    #     return ts.restart(np.array([self._state], dtype=np.int32))
    #
    # def _step(self, action):
    #
    #     if self._episode_ended:
    #         # The last action ended the episode. Ignore the current action and start
    #         # a new episode.
    #         return self.reset()
    #
    #     # Make sure episodes don't go on forever.
    #     if action == 1:
    #         self._episode_ended = True
    #     elif action == 0:
    #         new_card = np.random.randint(1, 11)
    #         self._state += new_card
    #     else:
    #         raise ValueError('`action` should be 0 or 1.')
    #
    #     if self._episode_ended or self._state >= 21:
    #         reward = self._state - 21 if self._state <= 21 else -21
    #         return ts.termination(np.array([self._state], dtype=np.int32), reward)
    #     else:
    #         return ts.transition(
    #             np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

    def __init__(self, tot_row, tot_col, goal_rewards=None, step_cost=-0.01, observation_step=5):

        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col

        # Originally agent was started as random, I changed to be deterministic ( [0.5, 0.5] -> [1, 0] )
        # self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/ self.action_space_size
        self.transition_matrix = np.eye(self.action_space_size)

        # from tf_agents.specs import array_spec
        # array_spec.BoundedArraySpec(shape=(tot_row, tot_col), dtype=np.int32, minimum=-1, maximum=4, name='Map')
        # Spec variables are needed to show the specification for some other variables
        self._map_spec = array_spec.BoundedArraySpec(shape=(tot_row, tot_col), dtype=np.int32, minimum=-1, maximum=4, name='map')
        self.state_matrix = np.zeros((tot_row, tot_col))  # Environmental Map of walls and goals

        # array_spec.BoundedArraySpec(shape=(observation_step, observation_step), dtype=np.int32, minimum=-1, maximum=4, name='Observation')
        self._observ_spec = array_spec.BoundedArraySpec(shape=(observation_step, observation_step), dtype=np.int32, minimum=-1, maximum=4, name='observation')

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        # array_spec.BoundedArraySpec(shape=(1, 1), dtype=np.int32, minimum=0, maximum=np.max(tot_row, tot_col), name='PlayerPosition')
        self.position = [0, 0]  # Indexes of Player position
        self.initial_position = [np.random.randint(tot_row), np.random.randint(tot_col)]

        # Set the reward for each goal A, B, C, D.
        # It could differ for each agent,
        # So, at the beginning of the game it sets for an agent individually
        if goal_rewards is None:
            goal_rewards = [1, 2, 4, 8]
        self.goal_rewards = goal_rewards

        # Set step cost in the environment
        # It could differ from experiment to experiment,
        # So, should be set at the beginning of the game
        self.step_cost = step_cost

        discount = 0.99
        step_type = "First"
        reward = 1
        observation = np.zeros((observation_step, observation_step))
        # tf_agents.trajectories.TimeStep(step_type, reward, discount, observation)
        self._current_time_step = (observation, reward, step_type, discount)

        super().__init__()

    def time_step_spec_grid_world(self):
        """Return time_step_spec."""
        return "A tuple of Observation, Reward, Step_Type and Discount. Example: " \
                  + str(np.shape(self._current_time_step)) + \
                  "\n\v * Observation is a part of a map seen by a player" \
                  "\n\v * Reward is a points the player received from the last move" \
                  "\n\v * Step_Type shows whether the move is \"First\", \"Mid\" or \"Last\"" \
                  "\n\v * Discount shows the decay of the next reward value"

    def map_spec(self):
        """Return observation_spec."""
        return self._map_spec

    def observation_spec(self):
        """Return observation_spec."""
        #return self._observ_spec
        # To fix compatibility with TensorFlow standard
        return self._map_spec

    def action_spec(self):
        """Return action_spec."""

        # Haven't decided which return would be more helpful
        # print("There are " + str(self.action_space_size) + " moves available. Put numbers from 0 to " \
        #         + str(self.action_space_size-1) + " to use an action for a step")
        return self._action_spec

    def _reset(self):
        """Return initial_time_step."""
        self.position = self.initial_position

        # In the future, it should output Observed map (7x7)
        return ts.restart(observation=np.array(self.state_matrix, dtype=np.int32))

    """
    According to Open AI principles applied to Gym package -
    Step function should:
        Do: make an action that agent wants in the environment
        Output:
            - New observation of the world (the whole world or limited section)
            - Collected reward after applying an agent's step
            - Status if the game is terminated or not (if the goal is reached - the game is done!)
    """
    def _step(self, action):
        """Apply action and return new time_step."""
        """ One step in the world.
                [observation, reward, done = env.step(action)]
                The robot moves one step in the world based on the action given.
                The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
                @return observation the position of the robot after the step
                @return reward the reward associated with the next state
                @return done True if the state is terminal
                """
        if  0 >= action >= self.action_space_size:
            raise ValueError('The action is not included in the action space.')

        # print("The Player's position is: " + str(self.position))
        # print("The Step() function is called with action: " + str(action))

        #************************************************
        #*** This code Will Not Work with TensorFlow standard!
        #************************************************
        # Based on the current action and the probability derived
        # from the transition model it chooses a new action to perform
        # if action_type == "random":
        #     # Picking randomly an action in accordance to the probabilities
        #     # Stored in transition_matrix
        #     action = np.random.choice(4, 1, p=self.transition_matrix[int(action), :])
        # elif action_type == "optimal":
        #     # Picking the action with highest probability
        #     # Stored in transition_matrix
        #     action = action
        # else:
        #     raise ValueError("The action_type is wrong!")

        # Check the boarders and
        # Move the player
        # Actions: 0 1 2 3 <-> UP RIGHT DOWN LEFT
        if action == 0 and self.position[0] > 0:
            new_position = [self.position[0] - 1, self.position[1]]
        elif action == 1 and self.position[1] < self.world_col-1:
            new_position = [self.position[0], self.position[1] + 1]
        elif action == 2 and self.position[0] < self.world_row-1:
            new_position = [self.position[0] + 1, self.position[1]]
        elif action == 3 and self.position[1] > 0:
            new_position = [self.position[0], self.position[1] - 1]
        else:
            # You hit the border - Terminate the game
            print("Player goes out of the borders")
            return ts.termination(np.array(self.state_matrix, dtype=np.int32), reward=-1)

        # Check if player has hit the wall on its move...
        hit_wall = self.state_matrix[new_position[0], new_position[1]] == -1

        # NOTE: Redundant check, however, reduces risk if bug appears
        # Check if the new position is a valid position
        # ! if you go to the wall - the move is ignored and the cost of move is calculated!
        if 0 <= new_position[0] < self.world_row:
            if 0 <= new_position[1] < self.world_col:
                if not hit_wall:
                    self.state_matrix[self.position[0], self.position[
                        1]] = 0  # Could be replaced with new object to save the trace of trajectory
                    self.position = new_position

        # to deal with variable visibility
        reward = 0

        # Return an occasion when the wall is hit
        if hit_wall:
            # Not False in case if the game didn't terminate on the goal cell
            done = bool(self.state_matrix[self.position[0], self.position[1]])
            reward = self.step_cost

            # In the future, it should output Observed map (7x7)
            return ts.transition(np.array(self.state_matrix, dtype=np.int32), reward=reward, discount=1)

        # Otherwise calculate the reward for according to a new cell
        match self.state_matrix[self.position[0], self.position[1]]:
            case 0:
                reward = self.step_cost  # Path
            case 1:
                reward = self.goal_rewards[0]  # Goal 1
            case 2:
                reward = self.goal_rewards[1]  # Goal 2
            case 3:
                reward = self.goal_rewards[2]  # Goal 3
            case 4:
                reward = self.goal_rewards[3]  # Goal 4

        # Done is True if the state is a terminal state
        done = bool(self.state_matrix[self.position[0], self.position[1]])

        # Terminate if goal is reached
        # Otherwise return TimeStamp standard output
        # Observation, Reward, Step_Type (?), discount
        if done:

            # In the future, it should output Observed map (7x7)
            return ts.termination(np.array(self.state_matrix, dtype=np.int32), reward)
        else:

            # In the future, it should output Observed map (7x7)
            return ts.transition(np.array(self.state_matrix, dtype=np.int32), reward=reward, discount=1)



    """
        Clears all the map, preparing for a new one
    """
    def clear(self):
        self.state_matrix = np.zeros((self.world_row, self.world_col))
        self.position = [np.random.randint(self.world_row), np.random.randint(self.world_col)]
        self.transition_matrix = np.eye(self.action_space_size)

    def setTransitionMatrix(self, transition_matrix):
        """
        The transition matrix here is intended as a matrix which has a line
        for each action and the element of the row are the probabilities to
        executes each action when a command is given. For example:
        [[0.55, 0.25, 0.10, 0.10]
         [0.25, 0.25, 0.25, 0.25]
         [0.30, 0.20, 0.40, 0.10]
         [0.10, 0.20, 0.10, 0.60]]
        This matrix defines the transition rules for all the 4 possible actions.
        The first row corresponds to the probabilities of executing each one of
        the 4 actions when the policy orders to the robot to go UP. In this case
        the transition model says that with a probability of 0.55 the robot will
        go UP, with a probaiblity of 0.25 RIGHT, 0.10 DOWN and 0.10 LEFT.
        """
        if transition_matrix.shape != self.transition_matrix.shape:
            raise ValueError('The shape of the two matrices must be the same.')
        self.transition_matrix = transition_matrix

    def setStateMatrix(self, state_matrix):
        """Set the obstacles, player and goals in the world.
        The input to the function is a matrix with the
        same size of the world
        -1 for states which are not walkable.
        +1 for terminal states [+1, +2, +3, +4] - for 4 different goals
         0 for all the walkable states (non-terminal)
        The following matrix represents the 4x3 world
        used in the series "dissecting reinforcement learning"
        [[+3,  -1,   0,   +1]
         [0,   -1,   0,   +4]
         [0,    0,   0,   +2]]
        """
        # print("State Matrix Shape: " + str(self.state_matrix.shape))
        # print("Your Shape: " + str(state_matrix.shape))
        if state_matrix.shape != self.state_matrix.shape:
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.state_matrix = state_matrix

    def setPosition(self):
        """ Set the position of a player and 4 Goals randomly
            But only on a walkable cells.
            ! Before using this method make sure you generated walls and put them
              like game.setStateMatrix(state_matrix)
        """

        # Next objects must be placed on the path: Player, Goal 1, Goal 2, Goal 3, Goal 4
        objectsToPlace = [10, 1, 2, 3, 4]
        for obj in objectsToPlace:
            randomRow = np.random.randint(self.world_row)
            randomCol = np.random.randint(self.world_col)
            # Ensure that the obj is placed on the path
            # The coordinates will be changed until it finds a clear cell
            while self.state_matrix[randomRow][randomCol] != 0:
                randomRow = np.random.randint(self.world_row)
                randomCol = np.random.randint(self.world_col)
                print(self.state_matrix[randomRow][randomCol])
            self.state_matrix[randomRow, randomCol] = obj    # Record obj position on the map
            if obj == 10:
                self.position = [randomRow, randomCol]
                self.initial_position = [randomRow, randomCol]

    def getWorldState(self):
        return self.state_matrix

    def getPlayerPosition(self):
        return self.position

    def render_map(self):
        """ Print the current world in the terminal.
        O           represents the player's position
        -           represents empty states.
        #           represents obstacles
        A, B, C, D  represent goals
        """
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):

                # Draw player
                if self.position == [row, col]: row_string += u" \u25CB " # u" \u25CC "

                # Draw walls, paths and goals
                else:
                    match self.state_matrix[row, col]:
                        # Wall
                        case -1:
                            row_string += ' # '
                        # Path
                        case 0:
                            row_string += ' - '
                        # Goal 1
                        case 1:
                            row_string += ' A '
                        # Goal 2
                        case 2:
                            row_string += ' B '
                        # Goal 3
                        case 3:
                            row_string += ' C '
                        # Goal 4
                        case 4:
                            row_string += ' D '

            row_string += '\n'
            graph += row_string
        print(graph)