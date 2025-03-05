import numpy as np


BOARD_WIDTH, BOARD_HEIGHT = 10, 20


def can_place(board, piece, x, y):
    """
    Check whether a given piece (2D numpy array) can be placed on the board
    with its top-left corner at (x, y). The placement is valid if:
      - Every nonzero cell in the piece is within the board bounds.
      - There is no collision (i.e. board cell is already nonzero).
    """
    piece_rows, piece_cols = piece.shape
    for i in range(piece_rows -1, -1, -1):
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


def getPieceDecision(
    board: np.array, piece: np.array, starting_x: int, rating_function
) -> dict[tuple, np.array]:
    """
    Returns the (rotation, horizontal translation) tuple corresponding to the best decision for the given board, active piece, and rating function. Active piece is assumed to start at (0, starting_x).
    """
    piece_size = len(piece)  # pieces are always square
    best_decision, best_rating = (), -100000000
    for r in range(-1, 3):
        rotated_piece = np.rot90(piece, r)
        for x_shift in range(-7, 7):
            # find the lowest position that this piece can be dropped with this x (if possible)
            # account for gravity – piece must fall by one for every x translation
            y = abs(x_shift)
            if not can_place(board, rotated_piece, x_shift + starting_x, y):
                continue
            y += 1
            while can_place(board, rotated_piece, x_shift + starting_x, y):
                y += 1
            y -= 1
            board[
                y : y + piece_size,
                x_shift + starting_x : x_shift + starting_x + piece_size,
            ] += rotated_piece
            curr_rating = rating_function(board)
            if curr_rating > best_rating:
                best_rating = curr_rating
                best_decision = (r, x_shift)
            board[
                y : y + piece_size,
                x_shift + starting_x : x_shift + starting_x + piece_size,
            ] -= rotated_piece
    return best_decision


def getCurrentBoardAndPiece(raw_board: np.array, active_mask: np.array):
    """
    Separates the board from the active tetromino. Also returns leftmost x coordinate of active tetromino
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
    return current_board, raw_board[:bottom, left:right], left
