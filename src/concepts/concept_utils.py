import numpy as np
from env import GoEnv, govars
from tqdm import tqdm
from typing import Tuple

def play_match(agents, board_size, concept_function, sample_ratio, binary=True, nn_format=False):
    go_env = GoEnv(board_size)
    go_env.reset()

    player_to_start = 1 if np.random.random() > 0.5 else 0
    current_player = player_to_start

    total_moves = board_size * board_size * 4
    moves = 0
    random_moves = False

    positive_cases = []
    negative_cases = []

    game_over = False

    prev_turn_state = np.zeros((board_size, board_size))
    temp_prev_turn_state = np.zeros((board_size, board_size))
    prev_opposing_state = np.zeros((board_size, board_size))

    while not game_over:
        if moves == total_moves:
            break

        state = go_env.canonical_state()

        valid_moves = go_env.valid_moves()

        state_copy = np.array([state[0], prev_turn_state, state[1], prev_opposing_state, np.full((board_size, board_size), current_player)])

        if random_moves:
            action = go_env.uniform_random_action()
        else:
            action, _ = agents[current_player].best_action(state_copy, valid_moves)

        _, _, game_over, _ = go_env.step(action)

        state_after_action = go_env.canonical_state()

        state_to_sample = np.array([state_after_action[1], state[0], state_after_action[0], state[1], np.full((board_size, board_size), current_player)])
            
        if np.random.random() < sample_ratio:
            if nn_format:
                pos = concept_function(state_to_sample)
            else:
                # TODO Might need to change the current player perspective
                pos = concept_function(state_after_action)
            
            
            if binary:
                if pos:
                    positive_cases.append(state_to_sample)
                elif not pos:
                    negative_cases.append(state_to_sample)
            else:
                positive_cases.append(state_to_sample)
                negative_cases.append(pos)

        moves += 1

        current_player = 1 - current_player
        # Update the previous state
        prev_turn_state = temp_prev_turn_state
        prev_opposing_state = state[0]
        temp_prev_turn_state = prev_opposing_state
    
    return positive_cases, negative_cases

def play_match_one_hot_concepts(agents, board_size, one_hot_encode_concepts):
    states = []
    one_hot_concepts = []

    tmp_states = []
    tmp_states_after_action = []
    tmp_turn = []

    go_env = GoEnv(board_size)
    go_env.reset()

    player_to_start = 1 if np.random.random() > 0.5 else 0
    current_player = player_to_start

    player_turn = 0

    total_moves = board_size * board_size * 4
    moves = 0
    random_moves = False

    game_over = False

    prev_turn_state = np.zeros((board_size, board_size))
    temp_prev_turn_state = np.zeros((board_size, board_size))
    prev_opposing_state = np.zeros((board_size, board_size))

    while not game_over:
        if moves == total_moves:
            break

        state = go_env.canonical_state()

        valid_moves = go_env.valid_moves()

        state_copy = np.array([state[0], prev_turn_state, state[1], prev_opposing_state, np.full((board_size, board_size), current_player)])

        if random_moves:
            action = go_env.uniform_random_action()
        else:
            action, _ = agents[current_player].best_action(state_copy, valid_moves)

        _, _, game_over, _ = go_env.step(action)

        state_after_action = go_env.canonical_state()

        state_to_sample = np.array([state_after_action[1], state[0], state_after_action[0], state[1], np.full((board_size, board_size), current_player)])

        tmp_states.append(state_copy)
        tmp_states_after_action.append(state_to_sample)
        tmp_turn.append(player_turn)
            
        moves += 1

        player_turn = 1 - player_turn

        current_player = 1 - current_player
        # Update the previous state
        prev_turn_state = temp_prev_turn_state
        prev_opposing_state = state[0]
        temp_prev_turn_state = prev_opposing_state

    winner = go_env.winner()
    
    for (state, state_after_action, turn) in zip(tmp_states, tmp_states_after_action, tmp_turn):
        if turn == govars.BLACK and winner == 1:
            outcome = 1
        elif turn == govars.WHITE and winner == -1:
            outcome = 1
        elif turn == govars.BLACK and winner == -1:
            outcome = -1
        elif turn == govars.WHITE and winner == 1:
            outcome = -1
        else:
            AssertionError("Invalid winner")

        states.append(state)
            
        # Target for the concept bottleneck outputlayer
        one_hot_concept = one_hot_encode_concepts(state_after_action, outcome)
        one_hot_concepts.append(one_hot_concept)

    return states, one_hot_concepts


def generate_static_concept_datasets(cases_to_sample, agents, board_size, concept_function, sample_ratio=0.8, nn_format=False, binary=True):

    positive_cases = []
    negative_cases = []

    if binary:
        positive_bar = tqdm(total=cases_to_sample, desc=f"Positive cases for concept '{concept_function.__name__}'")
    else:
        positive_bar = tqdm(total=cases_to_sample, desc=f"Continues cases for concept '{concept_function.__name__}'")

    while len(positive_cases) < cases_to_sample or len(negative_cases) < cases_to_sample:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                pos, neg = play_match([agents[i], agents[j]], board_size, concept_function, sample_ratio, binary, nn_format=nn_format)
                positive_cases.extend(pos)
                negative_cases.extend(neg)
                positive_bar.update(len(pos))

    positive_cases = positive_cases[:cases_to_sample]
    negative_cases = negative_cases[:cases_to_sample]

    return positive_cases, negative_cases

def generate_one_hot_concepts_dataset(cases_to_sample, agents, board_size, one_hot_encode_concepts):
    board_states = []
    one_hot_concepts = []

    bar = tqdm(total=cases_to_sample, desc="Generating concept datasets")

    while len(board_states) < cases_to_sample:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                s, c, = play_match_one_hot_concepts([agents[i], agents[j]], board_size, one_hot_encode_concepts)
                board_states.extend(s)
                one_hot_concepts.extend(c)
                bar.update(len(s))

    board_states = board_states[:cases_to_sample]
    one_hot_concepts = one_hot_concepts[:cases_to_sample]

    return board_states, one_hot_concepts

def convolve_filter(board_state: np.ndarray, concept_filter: np.ndarray, x=1, y=0, count_occurances=False) -> Tuple[bool, int]:
    """
    This function convolves a concept filter over a board state and returns True if the concept is present in the board state.
    If count_occurances is True, the function will also return the number of occurances of the concept in the board state.
    Returns the index of where the concept is present in the board state.
    
    Args:
        board_state: np.ndarray
        The current board state
        
        concept_filter: np.ndarray
        The concept filter to convolve over the board state
        
        x: int
        The value that represents the concept (e.g. 1 for stone)
        
        y: int
        The value that represents the concept (e.g. 0 for empty)
        
        count_occurances: bool
        If True, the function will return the number of occurances of the concept in the board state
        
        Returns:
        presence: bool
        True if the concept is present in the board state
        
        nr_of_occurances: int
        The number of occurances of the concept in the board state
    """

    # Count the number of 1's and 0's in the concept filter
    total_sim = 0
    for i in range(0, concept_filter.shape[0]):
        for j in range(0, concept_filter.shape[1]):
            if concept_filter[i, j] == x:
                total_sim += 1
            elif concept_filter[i, j] == y:
                total_sim += 1

    # Fist see of concept is present in current state
    stride = 1
    filter_size_wide = concept_filter.shape[0]
    filter_size_height = concept_filter.shape[1]
    presence = False
    nr_of_occurances = 0

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
                    if concept_filter[k, l] == x and current_area[k, l] == x:
                        total -= 1
                    elif concept_filter[k, l] == y and current_area[k, l] == y:
                        total -= 1
            if total == 0:
                presence = True
                nr_of_occurances += 1

                if not count_occurances:
                    return presence, nr_of_occurances
    
    return presence, nr_of_occurances

def filter_at_position(board_state: np.ndarray, concept_filter: np.ndarray, total_sim, x, y, i, j):
    """
    This function checks if a concept filter is present at a specific position in the board state.
    Returns True if the concept filter is present at the specified position.

    Args:
        board_state: np.ndarray
        The current board state

        concept_filter: np.ndarray
        The concept filter to convolve over the board state

        x: int
        The value that represents the concept (e.g. 1 for stone)

        y: int
        The value that represents the concept (e.g. 0 for empty)

        i: int
        The row index of the board state where the concept filter should be checked

        j: int
        The column index of the board state where the concept filter should be checked

        Returns:
        presence: bool
        True if the concept filter is present at the specified position
    """

    filter_size_wide = concept_filter.shape[0]
    filter_size_height = concept_filter.shape[1]
    current_area = board_state[i:i+filter_size_wide, j:j+filter_size_height]

    total = total_sim

    for k in range(0, filter_size_wide):
        for l in range(0, filter_size_height):
            if concept_filter[k, l] == x and current_area[k, l] == x:
                total -= 1
            elif concept_filter[k, l] == y and current_area[k, l] == y:
                total -= 1
    if total == 0:
        return True
    return False

def convolve_filter_all_positions(current_state: np.ndarray, prev_state: np.ndarray, concept_filter: np.ndarray, x=1, y=0):
    """
    This function convolves a concept filter over a board state and returns True if the concept is present in the current board state and not in the previous board state.
    If the concept is present in the current board_state and in the previous board_state, the function continues to find the next occurance of the concept in the current board_state and
    tests the concept in the prev board_state at the same position. This continues until the end of the board_state is reached.
    Returns true if the concept is present in the current board_state and not in the previous board_state.
    """

    # Count the number of 1's and 0's in the concept filter
    total_sim = 0
    for i in range(0, concept_filter.shape[0]):
        for j in range(0, concept_filter.shape[1]):
            if concept_filter[i, j] == x:
                total_sim += 1
            elif concept_filter[i, j] == y:
                total_sim += 1

    # Fist see of concept is present in current state
    stride = 1
    filter_size_wide = concept_filter.shape[0]
    filter_size_height = concept_filter.shape[1]

    # Check if the current state maches the 1's in the concept filter
    # -1's in the concept filter are ignored
    for i in range(0, current_state.shape[0]-2, stride):
        for j in range(0, current_state.shape[0]-2, stride):
            current_area = current_state[i:i+filter_size_wide, j:j+filter_size_height]
            if current_area.shape != concept_filter.shape:
                continue
            total = total_sim
            for k in range(0, filter_size_wide):
                for l in range(0, filter_size_height):
                    if concept_filter[k, l] == x and current_area[k, l] == x:
                        total -= 1
                    elif concept_filter[k, l] == y and current_area[k, l] == y:
                        total -= 1
            if total == 0:
                presence_prev = filter_at_position(prev_state, concept_filter, total_sim, x, y, i, j)
                if not presence_prev:
                    return True

    return False