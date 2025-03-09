import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any
import time, pstats

# import numba as nb


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
ROTATED_TETROMINOES = [[np.rot90(piece, r) for r in range(4)] for piece in TETROMINOES]


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


# Only the bottom pieces of the tetromino matter when checking if it can be placed
# Get tuple of lists containing lowest cell for each col in a tetromino
# (0, 0) is the top left cell in this tetromino
# Precompute this to save time
def get_bottom_cells(piece):
    xs, ys = [], []
    for x in range(piece.shape[1]):
        y = piece.shape[0] - 1
        while y >= 0 and piece[y, x] == 0:
            y -= 1
        if y >= 0:
            xs.append(x)
            ys.append(y)
    return xs, ys


ROTATED_TETROMINOES_BOTTOM_CELLS = []
for r in ROTATED_TETROMINOES:
    ROTATED_TETROMINOES_BOTTOM_CELLS.append([])
    for rp in r:
        ROTATED_TETROMINOES_BOTTOM_CELLS[-1].append(get_bottom_cells(rp))


# Directly read the bottom indices of the tetromino
def can_place(board, piece_xs, piece_ys, starting_x, starting_y):
    for i in range(len(piece_xs)):
        if board[starting_y + piece_ys[i], starting_x + piece_xs[i]] != 0:
            return False
    return True


def tryHardDrop(board, piece, piece_bottom_cells, x, starting_y):
    piece_xs, piece_ys = piece_bottom_cells
    y = starting_y
    if not can_place(board, piece_xs, piece_ys, x, y):
        return None
    y += 1
    while can_place(board, piece_xs, piece_ys, x, y):
        y += 1
    y -= 1
    new_board = board.copy()
    new_board[
        y : y + len(piece),
        x : x + len(piece),
    ] += piece
    return new_board


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
    # Precompute rotations of current pieces and their bottom cells
    rotated_pieces = []
    rotated_pieces_bottom_cells = []
    for piece in current_pieces:
        if piece is not None:
            rotated_pieces.append(ROTATED_TETROMINOES[piece[1, 1] - 2])
            rotated_pieces_bottom_cells.append(
                ROTATED_TETROMINOES_BOTTOM_CELLS[piece[1, 1] - 2]
            )
        else:
            rotated_pieces.append(None)
            rotated_pieces_bottom_cells.append(None)
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
                rp_bottom_cells = rotated_pieces_bottom_cells[candidate.api][r % 4]
                for x_shift in range(-7, 7):
                    starting_x = getStartingX(rp)
                    new_board = tryHardDrop(  # over half the total calculation time comes from this
                        candidate.board,
                        rp,
                        rp_bottom_cells,
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
