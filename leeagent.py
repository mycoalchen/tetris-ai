import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from utils import getPossibleBoards, BOARD_HEIGHT, BOARD_WIDTH, getCurrentBoardAndPiece
import numpy as np


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

    def getPieceDecision(self, board, currentPiece):
        """
        Returns the (rotation, horizontal translation) tuple maximizing the rating of the board after hard dropping currentPiece
        """
        possibleBoards = getPossibleBoards(board, currentPiece)
        best_decision = ()
        best_rating = -1000000
        for (r, x), (new_board, _) in possibleBoards.items():
            if self.rateBoard(new_board) > best_rating:
                best_rating = self.rateBoard(new_board)
                best_decision = (r, x)
        return best_decision


env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
agent = leeAgent()

J = 0
i = 0
for _ in range(10):
    env.reset()
    terminated = False
    while not terminated:
        if i == 4:
            current_board, current_piece = getCurrentBoardAndPiece(observation["board"], observation["active_tetromino_mask"])
            print(current_board)
            print(current_piece)
            print(agent.getPieceDecision(current_board, current_piece))
            # colHeights = leeAgent.getColHeights(board)
            # print(colHeights)
            # print("agg:", leeAgent.getAggregateHeight(colHeights))
            # print("complete lines:", leeAgent.getCompleteLines(board))
            # print("holes:", leeAgent.getHoles(board))
            # print("bumpiness:", leeAgent.getBumpiness(colHeights))
            exit(0)
        action = 5
        observation, reward, terminated, truncated, info = env.step(action)
        i += 1
        J += reward
J /= 10
print("J(pi)=", J)
