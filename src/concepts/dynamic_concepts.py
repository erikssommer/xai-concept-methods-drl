import numpy as np
from mcts import MCTS
from policy import ConvNet, FastPredictor, LiteModel
from env import GoEnv

"""
Dynamic concepts are concepts that are not fixed, but rather change over time in response to the environment.
"""


SIMULATION_COUNT = 500
BOARD_SIZE = 5
KOMI = 0.5
MOVE_CAP = BOARD_SIZE ** 2 * 4
C = 1
DETERMINISTIC_MOVES = 0
BEST_MODEL_PATH = f"../models/best_models/board_size_{BOARD_SIZE}/net_1000.keras"

neural_network = ConvNet(BOARD_SIZE, load_path=BEST_MODEL_PATH)

predictor = FastPredictor(LiteModel.from_keras_model(neural_network.model))

# 'both' or 'single'. Single means concepts where only one player is considered, both means concepts where both players are considered.
concept_type = None

def game_start_sequence():
    env = GoEnv(BOARD_SIZE, KOMI)
    env.reset()
    init_state = env.canonical_state()

    # Initialize the MCTS
    mcts = MCTS(init_state, SIMULATION_COUNT, BOARD_SIZE, MOVE_CAP, predictor, C, KOMI, DETERMINISTIC_MOVES)

    # Create the tree
    mcts.search()

    concept_type = 'single'

    # Subpar variations
    min_visit_count_diff = 0.1
    min_value_diff = 0.2

    if concept_type == 'single':
        t_rollout_depth = 10
    else:
        t_rollout_depth = 5
    
    maximum_depth_find_sub_rollout = t_rollout_depth - 5

    optimal_rollout_states = []
    subpar_rollout_states = []

    node = mcts.root
    while node.time_step < maximum_depth_find_sub_rollout:
        # Find the optimal next state given visit count and value
        highest_visit_count = 0
        optimal_child = None

        for child in node.children:
            if child.n_visit_count > highest_visit_count:
                highest_visit_count = child.n_visit_count
                optimal_child = child

        optimal_rollout_states.append(node.state)

        # Find the subpar next state with a minimum value difference of 0.20 and/or a visit count difference of 10% of the highest visit count
        sub_par_children = []
        for child in node.children:
            if child.n_visit_count < highest_visit_count * min_visit_count_diff or child.q_value() < optimal_child.q_value() - min_value_diff:
                sub_par_children.append(child)
        
        # Find the best subpar state to rollout
        best_subpar_state = None
        highest_sub_par_visit_count = 0

        for child in sub_par_children:
            if child.n_visit_count > highest_sub_par_visit_count:
                highest_sub_par_visit_count = child.n_visit_count
                best_subpar_state = child.state

        subpar_rollout_states.append(best_subpar_state)

        node = optimal_child
    
    return optimal_rollout_states, subpar_rollout_states






