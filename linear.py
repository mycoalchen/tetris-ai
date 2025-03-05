import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from utils import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    getCurrentBoardAndPiece,
    getPieceDecision,
)
import numpy as np
import cv2
import datetime
from lee import leeRating


def testLinearBot(ratingFunction, numTrials=10, render=False, progUpdates=True):

    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")

    J = 0
    for i in range(numTrials):
        curr_J = 0
        steps = 0
        t0 = datetime.datetime.now()
        calc_time = 0
        new_piece = True
        observation, _ = env.reset()
        terminated = False
        while not terminated:
            steps += 1
            # idk why but rendering doesn't work unless the next block is uncommented
            if render:
                env.render()
                a = None
                while a is None:
                    _ = cv2.waitKey(1)
                    if _ == ord(" "):
                        a = 1
            if new_piece:
                t1 = datetime.datetime.now()
                current_board, current_piece, current_piece_left_x = (
                    getCurrentBoardAndPiece(
                        observation["board"], observation["active_tetromino_mask"]
                    )
                )
                decision = getPieceDecision(
                    current_board, current_piece, current_piece_left_x, ratingFunction
                )
                if decision == ():
                    break
                r, x = decision
                calc_time += (datetime.datetime.now() - t1).total_seconds()
            new_piece = False
            match r, x:
                case 0, 0:
                    action = 5
                    new_piece = True
                case -1, _:
                    action = 4
                    r += 1
                case (1, _) | (2, _):
                    action = 3
                    r -= 1
                case _:
                    action = 0 if x < 0 else 1
                    x -= np.sign(x)
            observation, reward, terminated, truncated, info = env.step(action)
            curr_J += reward
            if progUpdates:
                if steps % 1000 == 0:
                    dt = (datetime.datetime.now() - t0).total_seconds()
                    print(
                        f"{steps} steps completed in {dt:.2f}s; average {1000 * dt/steps:.2f}ms per step"
                    )
                    print(
                        f"{calc_time:.2f}s spent calculating moves ({100 * calc_time/dt:.2f}%)"
                    )
        print(f"Trial {i} terminated with reward {curr_J}")
        J += curr_J
    print("J(pi)=", J / numTrials)


if __name__ == "__main__":
    testLinearBot(leeRating)
