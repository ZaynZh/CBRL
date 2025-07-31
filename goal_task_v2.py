import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import math

class GoalTask(gym.Env):
    """Custom 2D Field Environment"""

    def __init__(self, initial_position=None, goal=(10,10)):
        super(GoalTask, self).__init__()
        
        # Action space: Agent can move in x and y directions within [-1, 1]
        self.action_space = spaces.Discrete(4)

        # Task state space:
        self.state_space = spaces.Box(low=np.array([0, 0]), high=np.array([20, 20]), dtype=np.float32)
        
        # Observation space: Agent's position in the 2D plane
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1]), dtype=np.float32)
        
        # Initial positions (with Gaussian noise added)
        self.initial_positions = [(2, 2), (2, 18), (18, 2), (18, 18)]

        # Initial positions for test
        self.test_initial_position = initial_position

        # Agent's current position
        self.position = np.array([0.0, 0.0])

        #
        self.distance_to_goal = 0

        # List to store agent positions for trajectory plotting
        self.positions = []
        
        # Max steps for a episode
        self.max_steps = 1000

        self.episodic_length = 0

        # Global steps counter
        self.global_step_count = 0

        # Goal position
        self.goal = np.array(goal)

        self.episodic_return = 0

        self.step_size = 0.2


    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        # Randomly choose an initial position and add Gaussian noise N(0,1)
        if self.test_initial_position == None:
            initial_pos = np.array(self.initial_positions[np.random.randint(0, 4)])
            noise = np.random.normal(0, 1, size=2)  # Gaussian noise N(0,1)
            self.position = np.clip(initial_pos + noise, 0, 20)  # Ensure within bounds
        elif self.test_initial_position:
            initial_pos = np.array(self.test_initial_position)
            noise = np.array([(0.002,0.002),(0.002,-0.002),(-0.002,0.002),(-0.002,-0.002)])
            noise = noise[np.random.randint(0,4)]
            self.position = np.clip(initial_pos + noise, 0, 20)
        self.distance_to_goal = np.linalg.norm(self.position - self.goal)
        self.episodic_length = 0
        self.episodic_return = 0
        info = self._get_info()
        obs = self._get_obs()
        return obs, info

    def step(self, action):
        self.episodic_length += 1
        self.global_step_count += 1

        self.positions.append(self.position.copy())
        self.previous_distance_to_goal = np.linalg.norm(self.position - self.goal)

        # Discrete action to movement
        delta = np.array([0.0, 0.0])
        if action == 0:   # Up
            delta = np.array([0.0, self.step_size])
        elif action == 1: # Down
            delta = np.array([0.0, -self.step_size])
        elif action == 2: # Right
            delta = np.array([self.step_size, 0.0])
        elif action == 3: # Left
            delta = np.array([-self.step_size, 0.0])

        self.position += delta
        self.position = np.clip(self.position, self.state_space.low, self.state_space.high)
        out_of_bounds = np.any(self.position <= self.state_space.low) or np.any(self.position >= self.state_space.high)

        # Reward logic
        self.distance_to_goal = np.linalg.norm(self.position - self.goal)
        if self.distance_to_goal < 2:
            reward = 1
        elif out_of_bounds:
            reward = -0.02
        # elif self.previous_distance_to_goal - self.distance_to_goal > 0:
        #     reward = min(0.02, (self.previous_distance_to_goal - self.distance_to_goal) / 20)
        else:
            reward = 0

        self.episodic_return += reward
        obs = self._get_obs()
        info = self._get_info()
        done = self.distance_to_goal < 2 or self.episodic_length >= self.max_steps

        if done:
            self.positions.append(self.position.copy())
            self.reset()

        return obs, reward, done, False, info


    def render(self, mode='human'):
        """Renders the environment (optional)."""
        print(f'Agent position: {self.position}')

    def close(self):
        """Cleans up resources when the environment is closed."""
        pass

    def _get_obs(self):
        return np.array([self.position[0]/20, 
                         1-self.position[0]/20, 
                         self.position[1]/20, 
                         1-self.position[1]/20, 
                         1-self.distance_to_goal/(np.linalg.norm(self.state_space.high - self.state_space.low))])

    def _get_info(self):
        return{
            "episode":{
                "r":self.episodic_return,
                "l":self.episodic_length
            }
        }

    def plot_trajectory(self):
        """Plots the agent's trajectory using matplotlib."""
        positions = np.array(self.positions)
        plt.figure(figsize=(6, 6))
        plt.plot(positions[:, 0], positions[:, 1], marker='o', markersize=5, label='Agent trajectory')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Agent Trajectory in 2D Environment')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_trajectory(self):
        return np.array(self.positions)
    
    def change_goal(self, goal):
        self.goal = goal