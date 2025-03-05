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
    for i in range(piece_rows):
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


def getPossibleBoards(board: np.array, piece: np.array) -> dict[tuple, np.array]:
    """
    Return all possible future boards after placing this piece, with corresponding (rotation, horizontal translation) tuples, given that it starts at piece_x, piece_y.
    Returns dict mapping valid (rotation, horizontal translation) tuples to resulting future boards. Output is 0/1 board.
    Rotations are counterclockwise.
    """
    possible = {}
    for r in range(-1, 3):
        rotated_piece = np.rot90(piece, r)
        for x in range(-7, 7):
            # find the lowest position that this piece can be dropped with this x (if possible)
            # account for gravity – piece must fall by one for every x translation
            y = abs(x)
            if not can_place(board, rotated_piece, x + 7, y):
                continue
            y += 1
            while can_place(board, rotated_piece, x + 7, y):
                y += 1
            y -= 1
            new_board = board.copy()
            for i in range(4):
                for j in range(4):
                    if rotated_piece[i, j] != 0:
                        new_board[y + i, x + j + 7] = 9
            # Save the new board configuration with key (rotation, x translation)
            possible[(r, x)] = (new_board, rotated_piece)
    return possible


def getCurrentBoardAndPiece(raw_board, active_mask):
    """
    Separates the board from the active tetromino
    """
    current_board = raw_board - raw_board * active_mask
    current_piece = raw_board[0:4, 7:11]
    return current_board, current_piece
