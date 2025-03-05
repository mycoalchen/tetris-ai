import numpy as np
from collections import deque


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


def getBestHardDrop(current_board: np.array, active_piece: np.array, rating_function):
    """
    Returns the (rotation, horizontal translation) tuple corresponding to the best decision for the given board, active piece, and rating function. Active piece is assumed to start at (0, starting_x).
    """
    piece_size = len(active_piece)  # pieces are always square
    starting_x = getStartingX(active_piece)  # all pieces except I start at x = 8
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
    starting_x: int,
    rating_function,
):
    """
    Returns the best decision for the active piece, given the queue. Decision is either a (rotation, horizontal translation) tuple or True if best decision is to swap.
    Specifically, this finds the highest-rated sequence of decisions that clears piece_queue, and returns the first decision in that sequence.
    active_piece, held_piece, and piece_queue are assumed to contain SxS squares where S is the size of each piece. (2 for O, 4 for I, 3 for all others.)
    """
    if not piece_queue:
        return getBestHardDrop(current_board, active_piece, starting_x, rating_function)
    current_pieces = [held_piece, active_piece] + piece_queue
    # Precompute rotations of current pieces
    rotated_pieces = [
        [np.rot90(piece, r) for r in range(4)] for piece in current_pieces
    ]
    # BFS over all possible sequences of next hard drops
    # BFS states are composed of (current board, active piece index, held piece index, piece queue starting index, can swap, initial decision) tuples
    init_state = (current_board, 1, 0, 2, can_swap, None)
    bfs_queue = deque([init_state])
    best_initial_decision, best_end_rating = (), -1000000000
    visited_state_hashes = set()
    while bfs_queue:
        cb, api, hpi, pqi, cs, d0 = bfs_queue.pop()
        state_hash = hash((hash_board(cb), api, hpi, pqi, cs))
        if state_hash in visited_state_hashes:
            continue
        visited_state_hashes.add(state_hash)
        # check if terminal state
        if pqi == 2 + len(piece_queue):
            rating = rating_function(cb)
            if rating > best_end_rating:
                best_end_rating = rating
                best_initial_decision = d0
        # first try all the hard drops
        for r in range(-1, 3):
            rp = rotated_pieces[api][r % 4]
            for x_shift in range(-7, 7):
                pass
                y = abs(x_shift)
                piece_size = len(rp)
                sx = getStartingX(rp)  # all pieces except the I start at x = 8
                if not can_place(cb, rp, x_shift + sx, y):
                    continue
                y += 1
                while can_place(cb, rp, x_shift + sx, y):
                    y += 1
                y -= 1
                new_board = cb.copy()
                new_board[
                    y : y + piece_size,
                    x_shift + sx : x_shift + sx + piece_size,
                ] += rp
                # update d0 only if this is the first move in the sequence
                if not d0:
                    bfs_queue.appendleft((new_board, pqi, hpi, pqi + 1, cs, (r, x_shift)))
                else:
                    bfs_queue.appendleft((new_board, pqi, hpi, pqi + 1, cs, d0))
        # next try swapping
        if can_swap:
            pass
    return best_initial_decision
