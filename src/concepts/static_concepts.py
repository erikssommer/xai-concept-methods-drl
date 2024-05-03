import numpy as np
from env import gogame, govars
from .concept_utils import convolve_filter

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
    curr_pieces = game_state[0]
    opposing_pieces = game_state[1]

    concept_filter = np.array([
        [-1, 1, -1],
        [ 1, 0,  1],
        [-1, 1, -1]
    ])

    # Pad the current and previous state
    curr_pieces = np.pad(curr_pieces, 1, 'constant', constant_values=1)
    opposing_pieces = np.pad(opposing_pieces, 1, 'constant', constant_values=1)

    # Check if the current state maches the 1's in the concept filter
    # -1's in the concept filter are ignored
    presence_curr, _ = convolve_filter(curr_pieces, concept_filter)
    presence_opposing, _ = convolve_filter(opposing_pieces, concept_filter)

    return presence_curr or presence_opposing


def two_eyes(game_state) -> bool:
    """
    In the game of Go, an eye is a group of empty points surrounded by stones of a single color,
    such that no opposing stone can be placed in the group without being captured. Eyes are important
    because they allow a group of stones to have multiple liberties, making it more difficult for the
    opponent to capture the group.
    """
    black_pieces = game_state[0]
    white_pieces = game_state[1]

    concept_filter_0 = np.array([
        [ 1, 1, 1, 1, 1],
        [ 1, 0, 1, 0, 1],
        [ 1, 1, 1, 1, 1],
    ])

    concept_filter_45 = np.array([
        [ 1,  1,  1, -1, -1],
        [ 1,  0,  1, -1, -1],
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1,  0,  1],
        [-1, -1,  1,  1,  1],
    ])

    # Rotate the filters to get all possible orientations
    concept_filter_90 = np.rot90(concept_filter_0)
    concept_filter_135 = np.rot90(concept_filter_45)
    concept_filter_225 = np.rot90(concept_filter_135)
    concept_filter_270 = np.rot90(concept_filter_225)

    # Pad the current and previous state
    black_pieces = np.pad(black_pieces, 1, 'constant', constant_values=1)
    white_pieces = np.pad(white_pieces, 1, 'constant', constant_values=1)

    # Loop through all the filters and check if the concept is present in the current state and not in the previous state
    for concept_filter in [concept_filter_45, concept_filter_135, concept_filter_225, concept_filter_270, concept_filter_0, concept_filter_90]:
        presence_curr, _ = convolve_filter(black_pieces, concept_filter)
        presence_prev, _ = convolve_filter(white_pieces, concept_filter)
        if presence_curr or presence_prev:
            return True
        
    return False

def game_over(game_state) -> bool:
    """
    In the game of Go, game over is a situation where one player has no legal moves remaining and has lost the game.
    This is an equivalent concept to checkmate in chess.
    """
    # Check if the current player has any legal moves remaining
    legal_moves = gogame.valid_moves(game_state)

    # If there is only one legal move (pass), then the player has lost the game
    if np.sum(legal_moves) == 1:
        return True
    return False

def has_winning_move(game_state) -> bool:
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
            game_state_copy = np.copy(game_state)
            
            # Assert that the game state copy is the same as the original game state
            next_state = gogame.next_state(game_state_copy, i)

            # Check if the next player has any legal moves remaining
            next_player_legal_moves = gogame.valid_moves(next_state)
            if np.sum(next_player_legal_moves) == 1:
                return True

    return False

def capture_stones_threat(game_state) -> bool:
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
            game_state_copy = np.copy(game_state)

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

def play_center_in_opening(game_state) -> bool:
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

def ladder(game_state) -> bool:
    """
    Possible: Detect sequence of moves concepts
    In the game of Go, a ladder is a sequence of moves in which a group of stones is chased across
    the board. Ladders are important because they can be used to capture stones, or to defend stones
    from capture.
    """
    concept_filter = np.array([
        [ -1, -1,  0,  0],
        [ -1,  1, -1,  0],
        [ -1,  1,  1, -1],
        [  0, -1,  1,  0]
    ])

    # Rotate the filters to get all possible orientations
    concept_filter_90 = np.rot90(concept_filter)
    concept_filter_180 = np.rot90(concept_filter_90)
    concept_filter_270 = np.rot90(concept_filter_180)

    # Pad the current and previous state
    curr_state = game_state[0]
    opponent_state = game_state[1]

    # Merge the current and opponent state
    combined_state_1 = curr_state
    for i in range(len(opponent_state)):
        for j in range(len(opponent_state[i])):
            if opponent_state[i][j] == 1:
                combined_state_1[i][j] = -1

    combined_state_2 = opponent_state
    for i in range(len(curr_state)):
        for j in range(len(curr_state[i])):
            if curr_state[i][j] == 1:
                combined_state_2[i][j] = -1

    combined_state_1 = np.array(combined_state_1)
    combined_state_2 = np.array(combined_state_2)

    # Loop through all the filters and check if the concept is present in the current state and not in the previous state
    for concept_filter in [concept_filter, concept_filter_90, concept_filter_180, concept_filter_270]:
        presence_curr, _ = convolve_filter(combined_state_1, concept_filter, x=1, y=-1)
        if presence_curr:
            return True
        presence_curr, _ = convolve_filter(combined_state_2, concept_filter, x=1, y=-1)
        if presence_curr:
            return True

    return False

def number_of_own_stones(game_state) -> int:
    """
    Continuous concept (non-binary)
    In the game of Go, the number of stones that a player has on the board is an important concept
    because it can be used to determine the winner of the game.
    """
    return np.sum(game_state[0])

def difference_in_stones(game_state) -> int:
    """
    Continuous concept (non-binary)
    In the game of Go, the difference in the number of stones that each player has on the board is an important
    concept because it can be used to determine the winner of the game.
    """
    return np.sum(game_state[0]) - np.sum(game_state[1])

def number_of_eyes(game_state) -> int:
    """
    Continuous concept (non-binary)
    In the game of Go, the number of eyes that a group of stones has is an important concept because it can be used
    to determine if the group is alive or dead.
    """

    curr_pieces = game_state[0]
    opposing_pieces = game_state[1]

    concept_filter = np.array([
        [-1, 1, -1],
        [ 1, 0,  1],
        [-1, 1, -1]
    ])

    # Pad the current and previous state
    curr_pieces = np.pad(curr_pieces, 1, 'constant', constant_values=1)
    opposing_pieces = np.pad(opposing_pieces, 1, 'constant', constant_values=1)

    # Check if the current state maches the 1's in the concept filter
    # -1's in the concept filter are ignored
    _, nr_of_occurances_curr = convolve_filter(curr_pieces, concept_filter, count_occurances=True)
    _, nr_of_occurances_opposing = convolve_filter(opposing_pieces, concept_filter, count_occurances=True)

    return nr_of_occurances_curr + nr_of_occurances_opposing

def number_of_liberties(game_state) -> int:
    """
    Continuous concept (non-binary)
    In the game of Go, the number of liberties that a group of stones has is an important concept
    because it can be used to determine if the group is alive or dead.
    """
    return np.sum(gogame.liberties(game_state))

def number_of_legal_moves(game_state) -> int:
    """
    Continuous concept (non-binary)
    In the game of Go, the number of legal moves that a player has is an important concept because it can be used
    to determine if the player is in atari, or if the player has a winning move.
    """
    return np.sum(gogame.valid_moves(game_state))

def territory_score(game_state) -> int:
    """
    Continuous concept (non-binary)
    In the game of Go, territory score is a concept that describes the number of points that a player has surrounded.
    Territory score is important because it is used to determine the winner of the game.
    """
    black_area, _ = gogame.areas(game_state)

    return black_area


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
