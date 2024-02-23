"""
Concepts for JEM differs from the static concepts in 'static_concepts.py' by being present iff the move made creates the concept.
Meaning the concept is not present in the previous state, but is present in the current state.
This is done so that the agent can get a reward WHEN it does a move that creates a concept.
"""

import numpy as np
from env import gogame

from concepts import convolve_filter

def null(board_state: np.ndarray = None, desc="a generic move not tied to a strategy"):
    
    if board_state is None:
        return desc
    
    # Running all the other concepts to see if they are present
    if one_eye(board_state):
        return False
    if two_eyes(board_state):
        return False
    if capture_a_stone(board_state):
        return False
    if capture_group_of_stones(board_state):
        return False

    return True

def one_eye(board_state: np.ndarray = None, desc="creates one eye where the opponent cannot place a stone"):
    if board_state is None:
        return desc
    
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
    presence_curr = convolve_filter(curr_state, concept_filter)
    presence_prev = convolve_filter(prev_state, concept_filter)

    return presence_curr and not presence_prev

def two_eyes(board_state: np.ndarray = None, desc="creates two eyes resulting in forming a living group"):
    if board_state is None:
        return desc
    
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

    curr_state = board_state[0]
    prev_state = board_state[1]

    # Pad the current and previous state
    curr_state = np.pad(curr_state, 1, 'constant', constant_values=1)
    prev_state = np.pad(prev_state, 1, 'constant', constant_values=1)

    # Loop through all the filters and check if the concept is present in the current state and not in the previous state
    for concept_filter in [concept_filter_45, concept_filter_135, concept_filter_225, concept_filter_270, concept_filter_0, concept_filter_90]:
        presence_curr = convolve_filter(curr_state, concept_filter)
        presence_prev = convolve_filter(prev_state, concept_filter)
        if presence_curr and not presence_prev:
            return True
        
    return False

def capture_a_stone(board_state: np.ndarray = None, desc="captures one of the opponent's stones"):
    if board_state is None:
        return desc
    
    # If the current state has less stones than the previous state
    # then the concept is present
    curr_state_opponent = board_state[0]
    prev_state_opponent = board_state[1]

    # See if one of the 1´s in the previous state is a 0 in the current state
    for i in range(curr_state_opponent.shape[0]):
        for j in range(curr_state_opponent.shape[0]):
            if prev_state_opponent[i, j] == 1 and curr_state_opponent[i, j] == 0:
                return True
    
    return False

def capture_group_of_stones(board_state: np.ndarray = None, desc="captures a group of the oppontens stones"):
    if board_state is None:
        return desc
    
    # If the current state has less stones than the previous state
    # then the concept is present
    curr_state_opponent = board_state[0]
    prev_state_opponent = board_state[1]

    # See of the current state has more than one stone less than the previous state
    counter = 0
    for i in range(curr_state_opponent.shape[0]):
        for j in range(curr_state_opponent.shape[0]):
            if prev_state_opponent[i, j] == 1 and curr_state_opponent[i, j] == 0:
                counter += 1
                
    if counter > 1:
        return True
    
    return False


def area_advantage(board_state: np.ndarray = None, desc="provides area advantage by surrounding a larger area"):
    if board_state is None:
        return desc
    
    # If the current state has more stones than the previous state
    # then the concept is present
    black_area_curr, white_area_curr = gogame.areas_nn_format(board_state, 0, 2)
    black_area_prev, white_area_prev = gogame.areas_nn_format(board_state, 1, 3)

    # If the previous state has more black area than white area than the move does not provide area advantage
    if black_area_prev > white_area_prev:
        return False

    # If the current state has more black area than the previous state and more black area than white area
    if black_area_curr > black_area_prev and black_area_curr > white_area_curr:
        return True

    return False
    