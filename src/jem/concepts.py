"""
Concepts for JEM differs from the static concepts in 'static_concepts.py' by being present iff the move made creates the concept.
Meaning the concept is not present in the previous state, but is present in the current state.
This is done so that the agent can get a reward WHEN it does a move that creates a concept.
"""

import numpy as np
from env import gogame
from .explanations import Explanations
from typing import Optional, Tuple

from concepts import convolve_filter_all_positions

def concept_functions_to_use():
    """
    This function returns a list of all the concept functions to use.
    """
    return [
        null,
        play_center_in_opening,
        #two_eyes,
        one_eye,
        capture_a_stone,
        capture_group_of_stones,
        area_advantage
    ]

def null(board_state: np.ndarray = None, reward_shaping: bool = False) -> Tuple[bool, Optional[str], Optional[float]]:
    explanation, reward = Explanations.NULL.value

    if board_state is None:
        return explanation, reward

    # Running all the other concepts to see if they are present
    for concept_function in concept_functions_to_use():
        if concept_function == null:
            continue
        if concept_function(board_state):
            if reward_shaping:
                return False, explanation, reward
            else:
                return False
    
    if reward_shaping:
        return True, explanation, reward
    else:
        return True


def one_eye(board_state: np.ndarray = None, reward_shaping: bool = False) -> Tuple[bool, Optional[str], Optional[float]]:
    explanation, reward = Explanations.ONE_EYE.value
    if board_state is None:
        return explanation, reward

    concept_filter = np.array([
        [-1, 1, -1],
        [ 1, 0,  1],
        [-1, 1, -1]
    ])

    curr_state = board_state[0]
    prev_state = board_state[1]

    # Pad the current and previous state
    curr_state = np.pad(curr_state, 1, 'constant', constant_values=1)
    prev_state = np.pad(prev_state, 1, 'constant', constant_values=1)

    # Check if the current state maches the 1's in the concept filter
    # -1's in the concept filter are ignored
    presence = convolve_filter_all_positions(curr_state, prev_state, concept_filter)

    if reward_shaping:
        return presence, explanation, reward
    else:
        return presence


def two_eyes(board_state: np.ndarray = None, reward_shaping: bool = False) -> Tuple[bool, Optional[str], Optional[float]]:
    explanation, reward = Explanations.TWO_EYES.value
    if board_state is None:
        return explanation, reward

    concept_filter_0 = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
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

    filter_list = [concept_filter_45,
                   concept_filter_135,
                   concept_filter_225,
                   concept_filter_270,
                   concept_filter_0,
                   concept_filter_90]

    curr_state = board_state[0]
    prev_state = board_state[1]

    # Pad the current and previous state
    curr_state = np.pad(curr_state, 1, 'constant', constant_values=1)
    prev_state = np.pad(prev_state, 1, 'constant', constant_values=1)

    # Loop through all the filters and check if the concept is present in the current state and not in the previous state
    for concept_filter in filter_list:
        presence = convolve_filter_all_positions(curr_state, prev_state, concept_filter)
        if presence:
            if reward_shaping:
                return True, explanation, reward
            else:
                return True

    if reward_shaping:
        return False, explanation, reward
    else:
        return False


def capture_a_stone(board_state: np.ndarray = None, reward_shaping: bool = False) -> Tuple[bool, Optional[str], Optional[float]]:
    explanation, reward = Explanations.CAPTURE_A_STONE.value
    if board_state is None:
        return explanation, reward

    # If the current state has less stones than the previous state
    # then the concept is present
    curr_state_opponent = board_state[2]
    prev_state_opponent = board_state[3]

    # See if one of the 1´s in the previous state is a 0 in the current state
    for i in range(curr_state_opponent.shape[0]):
        for j in range(curr_state_opponent.shape[0]):
            if prev_state_opponent[i, j] == 1 and curr_state_opponent[i, j] == 0:
                if reward_shaping:
                    return True, explanation, reward
                else:
                    return True

    if reward_shaping:
        return False, explanation, reward
    else:
        return False


def capture_group_of_stones(board_state: np.ndarray = None, reward_shaping: bool = False) -> Tuple[bool, Optional[str], Optional[float]]:
    explanation, reward = Explanations.CAPTURE_GROUP_OF_STONES.value
    if board_state is None:
        return explanation, reward

    # If the current state has less stones than the previous state
    # then the concept is present
    curr_state_opponent = board_state[2]
    prev_state_opponent = board_state[3]

    # See of the current state has more than one stone less than the previous state
    counter = 0
    for i in range(curr_state_opponent.shape[0]):
        for j in range(curr_state_opponent.shape[0]):
            if prev_state_opponent[i, j] == 1 and curr_state_opponent[i, j] == 0:
                counter += 1

    if counter > 1:
        if reward_shaping:
            return True, explanation, reward
        else:
            return True

    if reward_shaping:
        return False, explanation, reward
    else:
        return False


def area_advantage(board_state: np.ndarray = None, reward_shaping: bool = False) -> Tuple[bool, Optional[str], Optional[float]]:
    explanation, reward = Explanations.AREA_ADVANTAGE.value
    if board_state is None:
        return explanation, reward

    # If the current state has more stones than the previous state
    # then the concept is present
    black_area_curr, white_area_curr = gogame.areas_nn_format(
        board_state, 0, 2)
    black_area_prev, white_area_prev = gogame.areas_nn_format(
        board_state, 1, 3)

    # If the previous state has more black area than white area than the move does not provide area advantage
    if black_area_prev > white_area_prev:
        if reward_shaping:
            return False, explanation, reward
        else:
            return False

    # If the current state has more black area than the previous state and more black area than white area
    if black_area_curr > black_area_prev and black_area_prev < white_area_prev:
        if reward_shaping:
            return True, explanation, reward
        else:
            return True

    if reward_shaping:
        return False, explanation, reward
    else:
        return False
    
def play_center_in_opening(board_state: np.ndarray = None, reward_shaping: bool = False) -> Tuple[bool, Optional[str], Optional[float]]:
    explanation, reward = Explanations.PLAY_CENTER_IN_OPENING.value
    if board_state is None:
        return explanation, reward
    
    # If the new stone in the current state is in the center of the board
    # Find the position of the new stone
    curr_state = board_state[0]
    prev_state = board_state[1]

    opponent_state = board_state[2]

    # Count the number of stones on the board
    stone_count = 0
    for i in range(curr_state.shape[0]):
        for j in range(curr_state.shape[0]):
            if curr_state[i, j] == 1:
                stone_count += 1

    for i in range(opponent_state.shape[0]):
        for j in range(opponent_state.shape[0]):
            if opponent_state[i, j] == 1:
                stone_count += 1
    
    # If the stone count is less than 10 then the move is in the opening
    if stone_count > 10:
        if reward_shaping:
            return False, explanation, reward
        else:
            return False


    # Find the position of the new stone
    for i in range(curr_state.shape[0]):
        for j in range(curr_state.shape[0]):
            if curr_state[i, j] == 1 and prev_state[i, j] == 0:
                if i >= 2 and i <= 4 and j >= 2 and j <= 4:
                    if reward_shaping:
                        return True, explanation, reward
                    else:
                        return True
                    
    if reward_shaping:
        return False, explanation, reward
    else:
        return False


