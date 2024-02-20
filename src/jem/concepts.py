"""
Concepts for JEM differs from the static concepts in 'static_concepts.py' by being present iff the move made creates the concept.
Meaning the concept is not present in the previous state, but is present in the current state.
This is done so that the agent can get a reward WHEN it does a move that creates a concept.
"""

import numpy as np

from concepts import convolve_filter

def one_eye(board_state: np.ndarray = None, name="creates one eye"):
    if board_state is None:
        return name
    
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

def two_eyes(board_state: np.ndarray = None, name="creates two eyes"):
    if board_state is None:
        return name
    
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


def center_dominance(board_state: np.ndarray = None, name="provides center dominance"):
    if board_state is None:
        return name
    
    # If the new stone is placed away from the border
    # then the concept is present
    curr_state = board_state[0]
    prev_state = board_state[1]

    presence_curr = np.sum(curr_state[1:-1, 1:-1]) > np.sum(prev_state[1:-1, 1:-1])

    return presence_curr


def area_advantage(board_state: np.ndarray = None, name="provides area advantage"):
    if board_state is None:
        return name
    
    # If the current state has more stones than the previous state
    # then the concept is present
    curr_state = board_state[0]
    prev_state = board_state[1]

    presence_curr = np.sum(curr_state) > np.sum(prev_state)

    return presence_curr
    