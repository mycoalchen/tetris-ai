import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any


# Use this in our heap while beam searching for future move sequences
# Beam search states are composed of (-1 * current board rating, current board, active piece index, held piece index, piece queue starting index, can swap, initial decision) tuples
@dataclass(order=True)
class LookaheadCandidate:
    negBoardRating: float
    board: Any = field(compare=False)
    api: int = field(compare=False)
    hpi: int = field(compare=False)
    pqi: int = field(compare=False)
    cs: bool = field(compare=False)
    d0: Any = field(compare=False)


BOARD_WIDTH, BOARD_HEIGHT = 10, 20


def can_place(board, piece, x, y):
    """
    Check whether a given piece (2D numpy array) can be placed on the board
    with its top-left corner at (x, y). The placement is valid if:
      - Every nonzero cell in the piece is within the board bounds.
      - There is no collision (i.e. board cell is already nonzero).
    """
    piece_rows, piece_cols = piece.shape
    for i in range(piece_rows - 1, -1, -1):
        for j in range(piece_cols):
            if piece[i, j] != 0:
                board_x = x + j
                board_y = y + i
                # Check boundaries.
                if (
                    board_x < 4
                    or board_x >= 4 + BOARD_WIDTH
                    or board_y < 0
                    or board_y >= BOARD_HEIGHT
                ):
                    return False
                # Check for collisions.
                if board[board_y, board_x] != 0:
                    return False
    return True


def getCurrentBoardAndPiece(raw_board: np.array, active_mask: np.array):
    """
    Separates the board from the active tetromino.
    """
    current_board = raw_board - raw_board * active_mask
    # find top left and bottom right corners of active mask
    x = 4
    while not active_mask[0, x]:
        x += 1
    left = x
    while active_mask[0, x]:
        x += 1
    right = x
    x -= 1
    y = 0
    while active_mask[y, x]:
        y += 1
    bottom = y
    return current_board, raw_board[:bottom, left:right]


def tryHardDrop(board: np.array, piece: np.array, starting_x: int, starting_y):
    # find the lowest position that this piece can be dropped with this x (if possible)
    # account for gravity – piece must fall by one for every x translation
    y = starting_y
    if not can_place(board, piece, starting_x, y):
        return None
    y += 1
    while can_place(board, piece, starting_x, y):
        y += 1
    y -= 1
    new_board = board.copy()
    new_board[
        y : y + len(piece),
        starting_x : starting_x + len(piece),
    ] += piece
    return new_board


def getBestHardDrop(current_board: np.array, active_piece: np.array, rating_function):
    """
    Returns the (rotation, horizontal translation) tuple corresponding to the best decision for the given board, active piece, and rating function. Active piece is assumed to start at (0, starting_x).
    """
    piece_size = len(active_piece)  # pieces are always square
    starting_x = getStartingX(active_piece)
    best_decision, best_rating = (), -100000000
    for r in range(-1, 3):
        rotated_piece = np.rot90(active_piece, r)
        for x_shift in range(-7, 7):
            # find the lowest position that this piece can be dropped with this x (if possible)
            # account for gravity – piece must fall by one for every x translation
            y = abs(x_shift)
            if not can_place(current_board, rotated_piece, x_shift + starting_x, y):
                continue
            y += 1
            while can_place(current_board, rotated_piece, x_shift + starting_x, y):
                y += 1
            y -= 1
            current_board[
                y : y + piece_size,
                x_shift + starting_x : x_shift + starting_x + piece_size,
            ] += rotated_piece
            curr_rating = rating_function(current_board)
            if curr_rating > best_rating:
                best_rating = curr_rating
                best_decision = (r, x_shift)
            current_board[
                y : y + piece_size,
                x_shift + starting_x : x_shift + starting_x + piece_size,
            ] -= rotated_piece
    return best_decision


TETROMINOES = [
    np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    ),  # I
    np.array([[1, 1], [1, 1]], dtype=np.uint8),  # O
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),  # T
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),  # S
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),  # Z
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),  # J
    np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),  # L
]


def read4x4(grid: np.array) -> np.array:
    """
    Read the tetromino from a 4x4 grid and return an SxS grid containing the result (S = 4 for I, 2 for O, 3 for all others)
    """
    for i in range(7):
        if np.any(grid[1:3, 1:3] == 2 + i):
            return TETROMINOES[i] * (2 + i)


def readQueue(queue: np.array, n) -> list[np.array]:
    """
    Read the first n elements of the queue
    """
    res = []
    for i in range(n):
        res.append(read4x4(queue[:, 4 * i : 4 * i + 4]))
    return res


def getStartingX(piece: np.array) -> int:
    """
    Get the initial x coordinate of the given piece
    Only I starts at x = 7; all other start at x = 8
    """
    return 7 if len(piece) == 4 else 8


def hash_board(board: np.array) -> int:
    return board.data.tobytes()


def getBestDecision(
    current_board: np.array,
    active_piece: np.array,
    held_piece: np.array,
    piece_queue: list[np.array],
    can_swap: bool,
    rating_function,
    num_beams=5,
):
    """
    Returns the best decision for the active piece, given the queue. Decision is either a (rotation, horizontal translation) tuple or True if best decision is to swap.
    Specifically, this finds the highest-rated sequence of decisions that clears piece_queue, and returns the first decision in that sequence.
    active_piece, held_piece, and piece_queue are assumed to contain SxS squares where S is the size of each piece. (2 for O, 4 for I, 3 for all others.)
    Maintains the top num_beams move sequences and prunes all others.
    """
    current_pieces = [held_piece, active_piece] + piece_queue
    # Precompute rotations of current pieces
    rotated_pieces = [
        [np.rot90(piece, r) for r in range(4)] for piece in current_pieces
    ]
    # Beam search over all possible sequences of next hard drops
    # Beam search states are composed of (-1 * current board rating, current board, active piece index, held piece index, piece queue starting index, can swap, initial decision) tuples
    init_state = LookaheadCandidate(0, current_board, 1, 0, 2, can_swap, None)
    candidates: list[LookaheadCandidate] = [init_state]
    terminals = []
    while candidates:
        new_candidates = []
        while candidates:
            candidate = candidates.pop()
            if candidate.pqi == len(current_pieces):
                terminals.append((-candidate.negBoardRating, candidate.d0))
                continue
            # try all the hard drops
            for r in range(-1, 3):
                rp = rotated_pieces[candidate.api][r % 4]
                for x_shift in range(-7, 7):
                    starting_x = getStartingX(rp)
                    new_board = tryHardDrop(candidate.board, rp, x_shift + starting_x, abs(x_shift))
                    if new_board is None:
                        continue
                    # update d0 only if this is the first move in the sequence
                    if not candidate.d0:
                        new_candidates.append(
                            LookaheadCandidate(
                                -rating_function(new_board),
                                new_board,
                                candidate.pqi,
                                candidate.hpi,
                                candidate.pqi + 1,
                                candidate.cs,
                                (r, x_shift),
                            )
                        )
                    else:
                        new_candidates.append(
                            LookaheadCandidate(
                                -rating_function(new_board),
                                new_board,
                                candidate.pqi,
                                candidate.hpi,
                                candidate.pqi + 1,
                                candidate.cs,
                                candidate.d0,
                            )
                        )
            # try swapping
            if candidate.cs:
                pass
        # prune all but the best num_beams sequences
        candidates = heapq.nsmallest(num_beams, new_candidates)
    best_rating, best_d0 = terminals.pop()
    while terminals:
        rating, d0 = terminals.pop()
        if rating > best_rating:
            best_rating, best_d0 = rating, d0
    return best_d0
