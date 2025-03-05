from utils import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
)
import numpy as np


weights = np.array([-0.510066, 0.760666, -0.35663, -0.184483])


def getColHeights(board):
    colHeights = np.zeros(BOARD_WIDTH)
    for x in range(4, 4 + BOARD_WIDTH):
        y = 0
        while y < BOARD_HEIGHT and not board[y, x]:
            y += 1
        colHeights[x - 4] = BOARD_HEIGHT - y
    return colHeights


def getAggregateHeight(colHeights):
    return np.sum(colHeights)


def getCompleteLines(board):
    completeLines = 0
    for y in range(BOARD_HEIGHT - 1, -1, -1):
        if np.all(board[y, :]):
            completeLines += 1
        if completeLines == 4:
            return 4
    return completeLines


def getHoles(board):
    holes = 0
    for x in range(4, 4 + BOARD_WIDTH):
        for y in range(BOARD_HEIGHT - 1, 0, -1):
            if not board[y, x] and board[y - 1, x]:
                holes += 1
    return holes


def getBumpiness(colHeights):
    return np.sum(np.abs(np.diff(colHeights)))


def leeRating(board):
    colHeights = getColHeights(board)
    return np.dot(
        weights,
        [
            getAggregateHeight(colHeights),
            getCompleteLines(board),
            getHoles(board),
            getBumpiness(colHeights),
        ],
    )
