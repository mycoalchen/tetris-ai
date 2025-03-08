import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any
import time, pstats


# for higher cprofile precision
def f8_alt(x):
    return "%14.9f" % x


pstats.f8 = f8_alt


# Use this in our heap while beam searching for future move sequences
# Beam search states are composed of (-1 * current board rating, current board, active piece starting y-coord, active piece index, held piece index, piece queue starting index, can swap, initial decision) tuples
@dataclass(order=True)
class LookaheadCandidate:
    negBoardRating: float
    board: Any = field(compare=False)
    apsy: int = field(compare=False)
    api: int = field(compare=False)
    hpi: int = field(compare=False)
    pqi: int = field(compare=False)
    cs: bool = field(compare=False)
    d0: Any = field(compare=False)
    decisions_rem: int = field(compare=False)


def hashCandidate(candidate: LookaheadCandidate) -> int:
    return hash(
        (
            candidate.board.tobytes(),
            candidate.apsy,
            candidate.api,
            candidate.hpi,
            candidate.pqi,
            candidate.cs,
            candidate.decisions_rem,
        )
    )


BOARD_WIDTH, BOARD_HEIGHT = 10, 20


def getCurrentBoardAndPiece(raw_board: np.array, active_mask: np.array):
    """
    Separates the board from the active tetromino. Also returns the top coordinate of the active piece.
    """
    current_board = raw_board - raw_board * active_mask
    # find top left and bottom right corners of active mask
    top = 0 if np.any(active_mask[0, :] == 1) else 1
    left = 7 if active_mask[top, 7] == 1 else 8
    x = left
    while active_mask[top, x]:
        x += 1
    right = x
    y = top
    while active_mask[y, left]:
        y += 1
    bottom = y
    return current_board, raw_board[top:bottom, left:right], top


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


def tryHardDrop(board: np.array, piece: np.array, starting_x: int, starting_y):
    # find the lowest position that this piece can be dropped with this x (if possible)
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


def get_slices(piece: np.array, x, y):
    height, width = piece.shape
    return tuple((slice(y, y + height), slice(x, x + width)))


def can_place_fast(board: np.array, piece: np.array, x: int, y: int):
    slices = get_slices(piece, x, y)
    board_subsection = board[slices]
    return board_subsection.shape == piece.shape and not np.any(
        board_subsection[piece > 0] > 0
    )


def tryHardDrop_fast(board, piece, starting_x, starting_y):  # ~40% faster than tryHardDrop
    y = starting_y
    if not can_place_fast(board, piece, starting_x, y):
        return None
    y += 1
    while can_place_fast(board, piece, starting_x, y):
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
            new_board = tryHardDrop_fast(
                current_board, rotated_piece, x_shift + starting_x, abs(x_shift)
            )
            if new_board is None:
                continue
            new_rating = rating_function(new_board)
            if new_rating > best_rating:
                best_rating, best_decision = new_rating, (r, x_shift)
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
    return TETROMINOES[grid[1, 1] - 2] * grid[1, 1] if grid[1, 1] > 1 else None


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


ROTATED_TETROMINOES = [[np.rot90(piece, r) for r in range(4)] for piece in TETROMINOES]


def getBestDecision(
    current_board: np.array,
    active_piece_starting_y: int,
    active_piece: np.array,
    held_piece: np.array,
    piece_queue: list[np.array],
    can_swap: bool,
    rating_function,
    num_beams,
    num_decisions,
):
    """
    Returns the best decision for the active piece, given the queue. Decision is either a (rotation, horizontal translation) tuple or (False, False) if best decision is to swap.
    Specifically, this finds the highest-rated sequence of num_decisions hard drops, and returns the first decision in that sequence.
    active_piece, held_piece, and piece_queue are assumed to contain SxS squares where S is the size of each piece. (2 for O, 4 for I, 3 for all others.)
    Maintains the top num_beams move sequences and prunes all others.
    """
    current_pieces = [held_piece, active_piece] + piece_queue
    # Precompute rotations of current pieces
    rotated_pieces = []
    for piece in current_pieces:
        if piece is not None:
            rotated_pieces.append(ROTATED_TETROMINOES[piece[1, 1] - 2])
        else:
            rotated_pieces.append(None)
    # Beam search over all possible sequences of next hard drops
    init_state = LookaheadCandidate(
        negBoardRating=-rating_function(current_board),
        board=current_board,
        apsy=active_piece_starting_y,
        api=1,
        hpi=0,
        pqi=2,
        cs=can_swap,
        d0=None,
        decisions_rem=num_decisions,
    )
    visited_states = set()
    candidates: list[LookaheadCandidate] = [init_state]
    terminals = []
    while candidates:
        new_candidates = []
        while candidates:
            candidate = candidates.pop()
            candidateHash = hashCandidate(candidate)
            if candidateHash in visited_states:
                continue
            visited_states.add(candidateHash)
            if candidate.decisions_rem == 0:
                terminals.append((-candidate.negBoardRating, candidate.d0))
                continue
            # try all the hard drops
            for r in range(-1, 3):
                rp = rotated_pieces[candidate.api][r % 4]
                for x_shift in range(-7, 7):
                    starting_x = getStartingX(rp)
                    new_board = tryHardDrop_fast(  # over half the total calculation time comes from this
                        candidate.board,
                        rp,
                        x_shift + starting_x,
                        candidate.apsy + abs(x_shift),
                    )
                    if new_board is None:
                        continue
                    # update d0 only if this is the first move in the sequence
                    new_candidates.append(
                        LookaheadCandidate(
                            negBoardRating=-rating_function(new_board),
                            board=new_board,
                            apsy=0,
                            api=candidate.pqi,
                            hpi=candidate.hpi,
                            pqi=candidate.pqi + 1,
                            cs=candidate.cs,
                            d0=(r, x_shift) if not candidate.d0 else candidate.d0,
                            decisions_rem=(candidate.decisions_rem - 1),
                        )
                    )
            # try swapping
            if candidate.cs:
                new_candidates.append(
                    LookaheadCandidate(
                        negBoardRating=candidate.negBoardRating,
                        board=candidate.board.copy(),
                        apsy=1,
                        api=candidate.hpi if held_piece is not None else candidate.pqi,
                        hpi=candidate.api,
                        pqi=(
                            candidate.pqi
                            if held_piece is not None
                            else candidate.pqi + 1
                        ),
                        cs=False,
                        d0=(False, False) if not candidate.d0 else candidate.d0,
                        decisions_rem=candidate.decisions_rem,
                    )
                )
        # prune all but the best num_beams sequences
        candidates = heapq.nsmallest(num_beams, new_candidates)
    best_rating, best_d0 = terminals.pop()
    while terminals:
        rating, d0 = terminals.pop()
        if rating > best_rating:
            best_rating, best_d0 = rating, d0
    return best_d0
