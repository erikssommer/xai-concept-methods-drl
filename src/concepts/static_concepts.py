import numpy as np
from env import gogame, govars

"""
Static concepts are concepts that can be determined from a single gamestate.

Note: Gamestates are represented as canonical states, meaning that the current player is always black.
"""

def random(game_state) -> bool:
    """
    Control concept that randomly returns True or False
    
    The random concept control ensures that what we are learning relates to specific concepts, 
    rather than simply separability of arbitrary points.
    """
    return np.random.random() < 0.5

def area_advantage(game_state) -> bool:
    """
    In the game of Go, area advantage is a concept that describes the difference between the number of
    points that a player has surrounded, and the number of points that their opponent has surrounded.
    Area advantage is important because it is used to determine the winner of the game.
    """
    black_area, white_area = gogame.areas(game_state)

    return black_area > white_area


def win_on_pass(game_state) -> bool:
    """
    In the game of Go, the game ends when both players pass in succession. The player with the most area advantage wins.
    """
    black_area, white_area = gogame.areas(game_state)

    # 4th index is the pass move
    prev_move_is_pass = bool(game_state[4][0][0] == 1)

    return prev_move_is_pass and black_area > white_area


def one_eye(game_state) -> bool:
    """
    In the game of Go, an eye is a empty point surrounded by stones of a single color
    """
    black_pieces = game_state[0]
    white_pieces = game_state[1]

    board = [[0 for _ in range(len(black_pieces))]
             for j in range(len(black_pieces))]

    # Combine black and white arrays into one array where black is 1 and white is -1
    for i in range(len(black_pieces)):
        for j in range(len(black_pieces[i])):
            if black_pieces[i][j] == 1:
                board[i][j] = 1
            elif white_pieces[i][j] == 1:
                board[i][j] = -1
            else:
                board[i][j] = 0

    # Using numpy pad function, pad the board with 1s around the edges
    black_board = np.pad(board, 1, 'constant', constant_values=1)

    # In the black frame array, check if there is a 0 surrounded by only 1s
    for i in range(len(black_board)):
        for j in range(len(black_board[i])):
            if black_board[i][j] == 0:
                # Check if the surrounding indexes are all 1s
                if black_board[i-1][j] == 1 and black_board[i+1][j] == 1 and black_board[i][j-1] == 1 and black_board[i][j+1] == 1:
                    return True

    # In the white frame array, check if there is a 0 surrounded by only -1s
    white_board = np.pad(board, 1, 'constant', constant_values=-1)

    for i in range(len(white_board)):
        for j in range(len(white_board[i])):
            if white_board[i][j] == 0:
                # Check if the surrounding indexes are all -1s
                if white_board[i-1][j] == -1 and white_board[i+1][j] == -1 and white_board[i][j-1] == -1 and white_board[i][j+1] == -1:
                    return True

    return False


def two_eyes(game_state):
    """
    In the game of Go, an eye is a group of empty points surrounded by stones of a single color,
    such that no opposing stone can be placed in the group without being captured. Eyes are important
    because they allow a group of stones to have multiple liberties, making it more difficult for the
    opponent to capture the group.
    """
    black_pieces = game_state[0]
    white_pieces = game_state[1]

    board = [[0 for _ in range(len(black_pieces))]
             for _ in range(len(black_pieces))]

    # Combine black and white arrays into one array where black is 1 and white is -1
    for i in range(len(black_pieces)):
        for j in range(len(black_pieces[i])):
            if black_pieces[i][j] == 1:
                board[i][j] = 1
            elif white_pieces[i][j] == 1:
                board[i][j] = -1
            else:
                board[i][j] = 0

    # Using numpy pad function, pad the board with 1s around the edges
    black_board = np.pad(board, 1, 'constant', constant_values=1)

    # In the black frame array, check if there is two 0s with one 1 between them and surrounded by only 1s
    for i in range(len(black_board)):
        for j in range(len(black_board[i])):
            if black_board[i][j] == 0:
                # Check if the surrounding indexes are all 1s
                if black_board[i-1][j] == 1 and black_board[i+1][j] == 1 and black_board[i][j-1] == 1 and black_board[i][j+1] == 1:
                    # Also need to test if the corners are 1s
                    if black_board[i-1][j-1] == 1 and black_board[i-1][j+1] == 1 and black_board[i+1][j-1] == 1 and black_board[i+1][j+1] == 1:
                        # Check if there is a 0 on the other side of the 1 to the left of the 0
                        if j - 2 >= 0 and black_board[i][j-2] == 0:
                            # Is covered by the right check
                            pass
                        # Check if there is a 0 on the other side of the 1 to the right of the 0
                        if j + 2 < len(black_board[i]) and black_board[i][j+2] == 0:
                            # Test if the surrounding indexes are all 1s
                            if black_board[i-1][j+2] == 1 and black_board[i+1][j+2] == 1 and black_board[i][j+3] == 1 and black_board[i][j+1] == 1:
                                # Also need to test if the corners are 1s
                                if black_board[i-1][j+3] == 1 and black_board[i+1][j+3] == 1 and black_board[i-1][j+1] == 1 and black_board[i+1][j+1] == 1:
                                    return True
                        # Check if there is a 0 on the other side of the 1 above the 0
                        if i - 2 >= 0 and black_board[i-2][j] == 0:
                            # Is covered by the bottom check
                            pass
                        # Check if there is a 0 on the other side of the 1 below the 0
                        if i+2 < len(black_board) and black_board[i+2][j] == 0:
                            # Test if the surrounding indexes are all 1s
                            if black_board[i-1][j] == 1 and black_board[i+3][j] == 1 and black_board[i][j+1] == 1 and black_board[i][j-1] == 1:
                                # Also need to test if the corners are 1s
                                if black_board[i-1][j-1] == 1 and black_board[i+3][j-1] == 1 and black_board[i-1][j+1] == 1 and black_board[i+3][j+1] == 1:
                                    return True

    white_board = np.pad(board, 1, 'constant', constant_values=-1)

    # In the white frame array, check if there is two 0s with one -1 between them and surrounded by only -1s
    for i in range(len(white_board)):
        for j in range(len(white_board[i])):
            if white_board[i][j] == 0:
                # Check if the surrounding indexes are all -1s
                if white_board[i-1][j] == -1 and white_board[i+1][j] == -1 and white_board[i][j-1] == -1 and white_board[i][j+1] == -1:
                    # Also need to test if the corners are -1s
                    if white_board[i-1][j-1] == -1 and white_board[i-1][j+1] == -1 and white_board[i+1][j-1] == -1 and white_board[i+1][j+1] == -1:
                        # Check if there is a 0 on the other side of the -1 to the left of the 0
                        if j - 2 >= 0 and white_board[i][j-2] == 0:
                            pass
                        # Check if there is a 0 on the other side of the -1 to the right of the 0
                        if j + 2 < len(black_board[i]) and white_board[i][j+2] == 0:
                            # Test if the surrounding indexes are all -1s
                            if white_board[i-1][j+2] == -1 and white_board[i+1][j+2] == -1 and white_board[i][j+3] == -1 and white_board[i][j+1] == -1:
                                # Also need to test if the corners are -1s
                                if white_board[i-1][j+3] == -1 and white_board[i+1][j+3] == -1 and white_board[i-1][j+1] == -1 and white_board[i+1][j+1] == -1:
                                    return True
                        # Check if there is a 0 on the other side of the -1 above the 0
                        if i - 2 >= 0 and white_board[i-2][j] == 0:
                            pass
                        # Check if there is a 0 on the other side of the -1 below the 0
                        if i+2 < len(black_board) and white_board[i+2][j] == 0:
                            # Test if the surrounding indexes are all -1s
                            if white_board[i-1][j] == -1 and white_board[i+3][j] == -1 and white_board[i][j+1] == -1 and white_board[i][j-1] == -1:
                                # Also need to test if the corners are -1s
                                if white_board[i-1][j-1] == -1 and white_board[i+3][j-1] == -1 and white_board[i-1][j+1] == -1 and white_board[i+3][j+1] == -1:
                                    return True

    return False

def tsumego(game_state):
    """
    In the game of Go, tsumego is a situation where one player has no legal moves remaining and has lost the game.
    This is an equivalent concept to checkmate in chess.
    """
    # Check if the current player has any legal moves remaining
    legal_moves = gogame.valid_moves(game_state)

    # If there is only one legal move (pass), then the player has lost the game
    if np.sum(legal_moves) == 1:
        return True
    return False

def has_winning_move(game_state):
    """
    The current player can play a move that will result in the next player having no legal moves remaining.
    This is an equivalent concept of having the ability to checkmate in chess.
    """
    # Loop through all legal moves and test if the next player has any legal moves remaining after making that move
    legal_moves = gogame.valid_moves(game_state)

    # If there is only one legal move (pass), then the player has lost the game
    if np.sum(legal_moves) == 1:
        return False
    
    for i in range(len(legal_moves)):
        if legal_moves[i] == 1:
            # Make a copy of the game state and play the move
            game_state_copy = game_state.copy()
            next_state = gogame.next_state(game_state_copy, i, canonical=True)

            # Check if the next player has any legal moves remaining
            next_player_legal_moves = gogame.valid_moves(next_state)
            if np.sum(next_player_legal_moves) == 1:
                return True

    return False

def capture_stones_threat(game_state):
    """
    In the game of Go, capture is a situation in which a group of stones is surrounded by the opponent's stones,
    and cannot escape. Capturing stones is an important concept because it can be used to gain an advantage over the opponent.
    """
    # Check if the current player has any legal moves remaining
    legal_moves = gogame.valid_moves(game_state)

    # If there is only one legal move (pass), then the player has lost the game
    if np.sum(legal_moves) == 1:
        return False
    
    # For every legal move, check if the move will capture any of the opponent's stones
    for i in range(len(legal_moves)):
        if legal_moves[i] == 1:
            # Make a copy of the game state and play the move
            game_state_copy = game_state.copy()

            # Count the number of black and white stones before the move
            opposing_pieces = game_state_copy[1]
            opposing_pieces_sum = np.sum(opposing_pieces)

            next_state = gogame.next_state(game_state_copy, i, canonical=True)
            
            # Count the number of black and white stones after the move
            opposing_pieces_after = next_state[0]
            opposing_pieces_sum_after = np.sum(opposing_pieces_after)

            # If the current player is black, check if the number of white stones has decreased
            if opposing_pieces_sum_after < opposing_pieces_sum:
                return True
                
    return False

def play_center_in_opening(game_state):
    """
    In Go, it is bad to play on the edge of the board during the opening phase of the game.
    """
    board_size = len(game_state[0])
    early_play_threshold = 9 if board_size == 5 else 13

    # Count the number of black and white stones
    black_pieces = game_state[0]
    white_pieces = game_state[1]
    black_pieces_sum = np.sum(black_pieces)
    white_pieces_sum = np.sum(white_pieces)
    total_pieces_sum = black_pieces_sum + white_pieces_sum

    if early_play_threshold >= total_pieces_sum and total_pieces_sum > 0:
        # Combine black and white arrays into one array
        combined_board = black_pieces
        for i in range(len(white_pieces)):
            for j in range(len(white_pieces[i])):
                if white_pieces[i][j] == 1:
                    combined_board[i][j] = 1
        
        combined_board = np.array(combined_board)

        # Check if the first or last row contains a black or white piece
        if np.sum(combined_board[0]) > 0 or np.sum(combined_board[-1]) > 0:
            return False
        # Check if the first or last column contains a black or white piece
        if np.sum(combined_board[:, 0]) > 0 or np.sum(combined_board[:, -1]) > 0:
            return False
        
        return True
    
    # Not in the opening phase of the game
    return None


def atari(game_state):
    """
    In the game of Go, atari is a situation in which a group of stones has only one liberty remaining.
    If the opponent places a stone on that liberty, the group will be captured. Atari is an important
    concept because it can be used to force the opponent to defend their stones, or to create a ko
    situation in which the opponent is not allowed to recapture the group immediately.
    """
    pass


def ko():
    """
    In the game of Go, ko is a situation in which a group of stones can be captured, but the opponent
    is not allowed to recapture the group immediately. Ko is an important concept because it can be
    used to force the opponent to defend their stones, or to create a ko situation in which the
    opponent is not allowed to recapture the group immediately.
    """
    pass


def ladder():
    """
    Possible: Detect sequence of moves concepts
    In the game of Go, a ladder is a sequence of moves in which a group of stones is chased across
    the board. Ladders are important because they can be used to capture stones, or to defend stones
    from capture.
    """
    pass


def seki():
    """
    In the game of Go, seki is a situation in which two groups of stones are adjacent to each other,
    and neither group can capture the other. Seki is important because it can be used to prevent the
    opponent from capturing a group of stones.
    """
    pass


def sente():
    """
    In the game of Go, sente is a situation in which a player is forced to respond to their opponent's
    move. Sente is important because it can be used to force the opponent to defend their stones, or
    to create a ko situation in which the opponent is not allowed to recapture the group immediately.
    """
    pass


def gote():
    """
    In the game of Go, gote is a situation in which a player is not forced to respond to their opponent's
    move. Gote is important because it can be used to force the opponent to defend their stones, or to
    create a ko situation in which the opponent is not allowed to recapture the group immediately.
    """
    pass


def sabaki():
    """
    In the game of Go, sabaki is a technique in which a player sacrifices stones in order to gain
    a better position elsewhere on the board. Sabaki is important because it can be used to gain
    an advantage over the opponent.
    """
    pass


def life_and_death():
    """
    A group of stones is said to be "alive" if it has at least two eyes, or if it cannot be captured by the opponent. 
    A group of stones is said to be "dead" if it can be captured by the opponent.
    """
    pass
