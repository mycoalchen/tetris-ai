import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
import cv2
import numpy as np


class leeAgent:
    
    def __init__(self):
        weights = np.array([-0.510066, 0.760666, -0.35663, -0.184483])
        

    # Given board, piece, holder, and queue, decide the best spot to place this piece or whether to hold it
    


env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
env = GroupedActionsObservations(env)

J = 0
for _ in range(10):
    env.reset()
    terminated = False
    while not terminated:
        action = 1
        observation, reward, terminated, truncated, info = env.step(action)
        print(env.action_space)
        print(env.legal_actions_mask)
        exit(0)
        J += reward
J /= 10
print('J(pi)=', J)