import numpy as np
from env import gogame, govars
import scipy.signal


def concept_area_advantage(game_state) -> bool:
    """
    In the game of Go, area advantage is a concept that describes the difference between the number of
    points that a player has surrounded, and the number of points that their opponent has surrounded.
    Area advantage is important because it is used to determine the winner of the game.
    """
    player = gogame.turn(game_state)
    black_area, white_area = gogame.areas(game_state)

    if player == govars.BLACK:
        return black_area > white_area
    else:
        return white_area > black_area


def concept_win_on_pass(game_state) -> bool:
    """
    In the game of Go, the game ends when both players pass in succession. The player with the most area advantage wins.
    """
    turn = gogame.turn(game_state)
    black_area, white_area = gogame.areas(game_state)

    # 4th index is the pass move
    prev_move_is_pass = game_state[4][0][0] == 1

    if turn == govars.BLACK:
        return prev_move_is_pass and black_area > white_area
    else:
        return prev_move_is_pass and white_area > black_area


def concept_one_eye(game_state) -> bool:
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


def concept_two_eyes(game_state):
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


def concept_atari(game_state):
    """
    In the game of Go, atari is a situation in which a group of stones has only one liberty remaining.
    If the opponent places a stone on that liberty, the group will be captured. Atari is an important
    concept because it can be used to force the opponent to defend their stones, or to create a ko
    situation in which the opponent is not allowed to recapture the group immediately.
    """
    pass


def concept_ko():
    """
    In the game of Go, ko is a situation in which a group of stones can be captured, but the opponent
    is not allowed to recapture the group immediately. Ko is an important concept because it can be
    used to force the opponent to defend their stones, or to create a ko situation in which the
    opponent is not allowed to recapture the group immediately.
    """
    pass


def concept_liberty():
    """
    In the game of Go, a liberty is an empty point adjacent to a group of stones. A group of stones
    has as many liberties as there are empty points adjacent to the group. Liberties are important
    because they allow a group of stones to remain on the board.
    """
    pass


def concept_territory():
    """
    In the game of Go, territory is an area of the board that is surrounded by stones of a single
    color. Territory is important because it is used to determine the winner of the game.
    """
    pass


def concept_ladder():
    """
    In the game of Go, a ladder is a sequence of moves in which a group of stones is chased across
    the board. Ladders are important because they can be used to capture stones, or to defend stones
    from capture.
    """
    pass


def concept_seki():
    """
    In the game of Go, seki is a situation in which two groups of stones are adjacent to each other,
    and neither group can capture the other. Seki is important because it can be used to prevent the
    opponent from capturing a group of stones.
    """
    pass


def concept_sente():
    """
    In the game of Go, sente is a situation in which a player is forced to respond to their opponent's
    move. Sente is important because it can be used to force the opponent to defend their stones, or
    to create a ko situation in which the opponent is not allowed to recapture the group immediately.
    """
    pass


def concept_gote():
    """
    In the game of Go, gote is a situation in which a player is not forced to respond to their opponent's
    move. Gote is important because it can be used to force the opponent to defend their stones, or to
    create a ko situation in which the opponent is not allowed to recapture the group immediately.
    """
    pass


def concept_sabaki():
    """
    In the game of Go, sabaki is a technique in which a player sacrifices stones in order to gain
    a better position elsewhere on the board. Sabaki is important because it can be used to gain
    an advantage over the opponent.
    """
    pass


def concept_life_and_death():
    """
    A group of stones is said to be "alive" if it has at least two eyes, or if it cannot be captured by the opponent. 
    A group of stones is said to be "dead" if it can be captured by the opponent.
    """
    pass
