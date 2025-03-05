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


class leeAgent:

    def __init__(self):
        self.weights = np.array([-0.510066, 0.760666, -0.35663, -0.184483])

    def getColHeights(self, board):
        colHeights = [0] * BOARD_WIDTH
        for x in range(4, 4 + BOARD_WIDTH):
            y = 0
            while y < BOARD_HEIGHT and not board[y, x]:
                y += 1
            colHeights[x - 4] = BOARD_HEIGHT - y
        return colHeights

    def getAggregateHeight(self, colHeights):
        return np.sum(colHeights)

    def getCompleteLines(self, board):
        completeLines = 0
        for y in range(BOARD_HEIGHT):
            if np.all(board[y, :]):
                completeLines += 1
            if completeLines == 4:
                return 4
        return completeLines

    def getHoles(self, board):
        holes = 0
        for x in range(4, 4 + BOARD_WIDTH):
            for y in range(BOARD_HEIGHT - 1, 0, -1):
                if not board[y, x] and board[y - 1, x]:
                    holes += 1
        return holes

    def getBumpiness(self, colHeights):
        return np.sum(np.abs(np.diff(colHeights)))

    def rateBoard(self, board):
        colHeights = self.getColHeights(board)
        return np.dot(
            self.weights,
            [
                self.getAggregateHeight(colHeights),
                self.getCompleteLines(board),
                self.getHoles(board),
                self.getBumpiness(colHeights),
            ],
        )


if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris")
    agent = leeAgent()

    J = 0
    i = 0
    prev1000 = 0
    t0 = datetime.datetime.now()
    calcTime = 0
    new_piece = True
    observation, _ = env.reset()
    terminated = False
    while not terminated:
        i += 1
        # idk why but rendering doesn't work unless the next block is uncommented
        # env.render()
        # a = None
        # while a is None:
        #     _ = cv2.waitKey(1)
        #     if _ == ord(" "):
        #         a = 1
        if new_piece:
            t1 = datetime.datetime.now()
            current_board, current_piece, current_piece_left_x = (
                getCurrentBoardAndPiece(
                    observation["board"], observation["active_tetromino_mask"]
                )
            )
            decision = getPieceDecision(
                current_board, current_piece, current_piece_left_x, agent.rateBoard
            )
            if decision == ():
                break
            r, x = decision
            calcTime += (datetime.datetime.now() - t1).total_seconds()
        new_piece = False
        match r, x:
            case 0, 0:
                action = 5
                new_piece = True
            # np.rot90 is counterclockwise
            # r = -1 means rotate clockwise
            # action 4 is rotate clockwise (Tetris mappings are flipped)
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
        J += reward
        if J > prev1000:
            print(J)
            prev1000 += 1000
        if i % 1000 == 0:
            dt = (datetime.datetime.now() - t0).total_seconds()
            print(
                f"{i} iterations completed in {dt:.2f}s; average {1000 * dt/i:.2f}ms per iteration"
            )
            print(f"{calcTime:.2f}s spent calculating moves ({100 * calcTime/dt:.2f}%)")
    print("J(pi)=", J)
