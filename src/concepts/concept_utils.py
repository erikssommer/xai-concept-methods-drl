import numpy as np
import env
from tqdm import tqdm

def play_match(agents, board_size, concept_function, nn_format=False):
    go_env = env.GoEnv(board_size)
    go_env.reset()

    player_to_start = 1 if np.random.random() > 0.5 else 0
    current_player = player_to_start

    total_moves = board_size * board_size * 4
    moves = 0
    random_moves = False

    positive_cases = []
    negative_cases = []

    sample_ratio = 0.8

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


def generate_static_concept_datasets(cases_to_sample, agents, board_size, concept_function, nn_format=False):

    positive_cases = []
    negative_cases = []

    positive_bar = tqdm(total=cases_to_sample, desc="Positive cases")

    while len(positive_cases) < cases_to_sample or len(negative_cases) < cases_to_sample:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                pos, neg = play_match([agents[i], agents[j]], board_size, concept_function, nn_format=nn_format)

                positive_cases.extend(pos)
                negative_cases.extend(neg)

                positive_bar.update(len(pos))

    positive_cases = positive_cases[:cases_to_sample]
    negative_cases = negative_cases[:cases_to_sample]

    return positive_cases, negative_cases