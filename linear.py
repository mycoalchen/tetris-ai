import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from utils import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    getCurrentBoardAndPiece,
    getBestDecision,
    readQueue,
    read4x4,
    getStartingX,
)
import numpy as np
import cv2
import time
from lee import leeRating


def testLinearBot(
    ratingFunction,
    lookahead=2,
    num_beams=5,
    numTrials=10,
    render=False,
    jUpdates=True,
    stepUpdates=False,
    allow_swaps=True,
):

    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", height=BOARD_HEIGHT)

    J = 0
    for i in range(numTrials):
        curr_J, prev_J1000 = 0, 0
        steps, moves = 0, 0
        t0 = time.time()
        calc_time = 0
        new_piece, can_swap = True, allow_swaps
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
                t1 = time.time()
                moves += 1
                current_board, active_piece, active_piece_starting_y = (
                    getCurrentBoardAndPiece(
                        observation["board"], observation["active_tetromino_mask"]
                    )
                )
                queue = readQueue(observation["queue"], lookahead + 1)
                decision = getBestDecision(
                    current_board=current_board,
                    active_piece_starting_y=active_piece_starting_y,
                    active_piece=active_piece,
                    held_piece=read4x4(observation["holder"]),
                    piece_queue=queue,
                    can_swap=can_swap,
                    rating_function=ratingFunction,
                    num_beams=num_beams,
                    num_decisions=lookahead + 1,
                )
                new_piece = False
                if decision == ():
                    break
                r, x = decision
                calc_time += time.time() - t1
            match r, x:
                case False, False:
                    action = 6
                    new_piece = True
                    can_swap = False
                case 0, 0:
                    action = 5
                    new_piece = True
                    can_swap = allow_swaps
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
            if stepUpdates:
                if steps % 100 == 0:
                    dt = time.time() - t0
                    print(
                        f"{steps} steps completed in {dt:.2f}s; average {1000 * dt/steps:.2f}ms per step"
                    )
                    print(
                        f"{moves} moves completed in {dt:.2f}s; average {1000 * dt/moves:.2f}ms per move"
                    )
                    print(
                        f"{calc_time:.2f}s spent calculating moves ({100 * calc_time/dt:.2f}%)"
                    )
            if jUpdates:
                if curr_J - curr_J % 1000 > prev_J1000:
                    print(f"Reached J = {curr_J} in {(time.time() - t0):.2f}s")
                    prev_J1000 = curr_J - curr_J % 1000
        print(f"Trial {i} terminated with reward {curr_J}")
        J += curr_J
    print("J(pi)=", J / numTrials)


if __name__ == "__main__":
    lookahead = 1
    num_beams = 5
    allow_swaps = True
    print("lookahead:", lookahead)
    print("num_beams:", num_beams)
    print("allow_swaps:", allow_swaps)
    print("board height:", BOARD_HEIGHT)
    testLinearBot(
        leeRating,
        render=False,
        jUpdates=True,
        stepUpdates=False,
        numTrials=5,
        lookahead=lookahead,
        num_beams=num_beams,
        allow_swaps=allow_swaps,
    )
