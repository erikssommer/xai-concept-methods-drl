"""
Concepts for JEM differs from the static concepts in 'static_concepts.py' by being present iff the move made creates the concept.
Meaning the concept is not present in the previous state, but is present in the current state.
This is done so that the agent can get a reward WHEN it does a move that creates a concept.
"""

import numpy as np

def one_eye(board_state: np.ndarray = None, name="creates an eye"):
    if board_state is None:
        return name
    
    concept_filter = np.array([
        [-1, 1, -1],
        [1, 0, 1],
        [-1, 1, -1]
    ])

    curr_state = board_state[0]
    prev_state = board_state[1]

    # Pad the current and previous state
    curr_state = np.pad(curr_state, 1, 'constant', constant_values=1)
    prev_state = np.pad(prev_state, 1, 'constant', constant_values=1)

    # Check if the current state maches the 1's in the concept filter
    # -1's in the concept filter are ignored
    presence_curr = __convolve(curr_state, concept_filter, 5)
    presence_prev = __convolve(prev_state, concept_filter, 5)

    return presence_curr and not presence_prev

def two_eyes(board_state: np.ndarray = None, name="creates an two eyes"):
    if board_state is None:
        return name
    
    concept_filter = np.array([
        [-1, 1, -1, 1, -1],
        [1, 0, 1, 0, 1],
        [-1, 1, -1, 1, -1],
    ])

    curr_state = board_state[0]
    prev_state = board_state[1]

    # Pad the current and previous state
    curr_state = np.pad(curr_state, 1, 'constant', constant_values=1)
    prev_state = np.pad(prev_state, 1, 'constant', constant_values=1)

    presence_curr = __convolve(curr_state, concept_filter, 9)
    presence_prev = __convolve(prev_state, concept_filter, 9)

    return presence_curr and not presence_prev

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


def __convolve(board_state: np.ndarray, concept_filter, total_sim=0):
    # Fist see of concept is present in current state
    stride = 1
    filter_size_wide = concept_filter.shape[0]
    filter_size_height = concept_filter.shape[1]
    presence = False

    # Check if the current state maches the 1's in the concept filter
    # -1's in the concept filter are ignored
    for i in range(0, board_state.shape[0]-2, stride):
        for j in range(0, board_state.shape[0]-2, stride):
            current_area = board_state[i:i+filter_size_wide, j:j+filter_size_height]
            if current_area.shape != concept_filter.shape:
                continue
            total = total_sim
            for k in range(0, filter_size_wide):
                for l in range(0, filter_size_height):
                    if concept_filter[k, l] == 1 and current_area[k, l] == 1:
                        total -= 1
                    elif concept_filter[k, l] == 0:
                        total -= 1
            if total == 0:
                presence = True
                break
        if presence:
            break
    
    return presence
    