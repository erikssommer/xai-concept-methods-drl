import numpy as np
import env
from tqdm import tqdm

def play_match(agents, board_size, concept_function, sample_ratio, nn_format=False):
    go_env = env.GoEnv(board_size)
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

        if current_player == 0:
            state_copy = np.array([state[0], prev_turn_state, state[1], prev_opposing_state, np.zeros((board_size, board_size))])
        else:
            state_copy = np.array([state[0], prev_turn_state, state[1], prev_opposing_state, np.ones((board_size, board_size))])
    
        if np.random.random() < sample_ratio:
            if nn_format:
                pos = concept_function(state_copy)
            else:
                pos = concept_function(state)
            
            if pos:
                positive_cases.append(state_copy)
            elif not pos:
                negative_cases.append(state_copy)

        if random_moves:
            action = go_env.uniform_random_action()
        else:
            action, _ = agents[current_player].best_action(state_copy, valid_moves)

        _, _, game_over, _ = go_env.step(action)

        moves += 1

        current_player = 1 - current_player
        # Update the previous state
        prev_turn_state = temp_prev_turn_state
        prev_opposing_state = state[0]
        temp_prev_turn_state = prev_opposing_state
    
    return positive_cases, negative_cases


def generate_static_concept_datasets(cases_to_sample, agents, board_size, concept_function, sample_ratio=0.8, nn_format=False):

    positive_cases = []
    negative_cases = []

    positive_bar = tqdm(total=cases_to_sample, desc=f"Positive cases for concept '{concept_function.__name__}'")

    while len(positive_cases) < cases_to_sample or len(negative_cases) < cases_to_sample:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                pos, neg = play_match([agents[i], agents[j]], board_size, concept_function, sample_ratio, nn_format=nn_format)

                positive_cases.extend(pos)
                negative_cases.extend(neg)

                positive_bar.update(len(pos))

    positive_cases = positive_cases[:cases_to_sample]
    negative_cases = negative_cases[:cases_to_sample]

    return positive_cases, negative_cases

def convolve_filter(board_state: np.ndarray, concept_filter: np.ndarray, x=1, y=0):

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
                break
        if presence:
            break
    
    return presence