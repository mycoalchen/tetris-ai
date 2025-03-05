import gymnasium as gym
from tetris_gymnasium.envs import Tetris
import cv2
from utils import getPossibleBoards

env = gym.make("tetris_gymnasium/Tetris", render_mode="human")

J = 0
i = 0
for _ in range(10):
    env.reset()
    terminated = False
    while not terminated:
        if i == 4:
            current_board = (
                observation["board"]
                - observation["board"] * observation["active_tetromino_mask"]
            )
            current_piece = observation["board"][0:4, 7:11]
            print(observation["board"])
            print(current_piece)
            possible = getPossibleBoards(current_piece, current_board)
            for tup, (board, rotated_piece) in possible.items():
                print(tup)
                print(rotated_piece)
                print(board)
        action = 5
        observation, reward, terminated, truncated, info = env.step(action)
        i += 1
        J += reward
J /= 10
print("J(pi)=", J)
