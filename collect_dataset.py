import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from utils import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    getCurrentBoardAndPiece,
    getBestDecision,
    readQueue,
    read4x4,
)
import numpy as np
import cv2
import time
from lee import leeRating


def collectSamples(
    num_samples,
    ratingFunction,
    lookahead=0,
    num_beams=5,
    allow_swaps=False,
):
    t0, samples_collected = time.time(), 0
    env = gym.make("tetris_gymnasium/Tetris")
    observations = {
        k: [None for _ in range(num_samples)]
        for k in ["board", "active_tetromino_mask", "queue", "holder"]
    }
    while samples_collected < num_samples:
        step_num = 0
        new_piece, can_swap = True, allow_swaps
        observation, _ = env.reset()
        terminated = False
        while not terminated and samples_collected < num_samples:
            if new_piece:
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
            step_num += 1
            # only collect samples after step_num > 100 to reduce bias towards earlier game states
            if step_num > 100:
                for k in observation:
                    observations[k][samples_collected] = observation[k]
                samples_collected += 1
            if samples_collected > 0 and samples_collected % 5_000 == 0:
                print(
                    f"Collected {samples_collected} samples in {(time.time() - t0):.2f}s"
                )
    for k in observations:
        observations[k] = np.stack(observations[k])
    np.savez("states_data", **observations)


if __name__ == "__main__":
    lookahead = 0
    num_beams = 5
    allow_swaps = False
    N = 500_000
    print("lookahead:", lookahead)
    print("num_beams:", num_beams)
    print("allow_swaps:", allow_swaps)
    print("N:", N)
    collectSamples(
        num_samples=N,
        ratingFunction=leeRating,
        lookahead=lookahead,
        num_beams=num_beams,
        allow_swaps=allow_swaps,
    )
