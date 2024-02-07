from mcts import MCTS
from mcts import Node
from policy import ConvNet, FastPredictor, LiteModel
import env

"""
Dynamic concepts are concepts that are not fixed, but rather change over time in response to the environment.
"""

class DynamicConcepts:
    def __init__(self, init_state, simulations, board_size, concept_type_single, path, move_cap=100):
        # 'both' or 'single'. Single means concepts where only one player is considered, both means concepts where both players are considered.
        self.concept_type_single = concept_type_single
        self.board_size = board_size

        neural_network = ConvNet(board_size, load_path=path)

        predictor = FastPredictor(LiteModel.from_keras_model(neural_network.model))

        # Initialize the MCTS
        self.mcts = MCTS(init_state, simulations, board_size, move_cap, predictor)

    def generate_cases(self):
        # Create the tree
        self.mcts.search()

        # Subpar variations
        min_visit_count_diff = 0.1
        min_value_diff = 0.2

        if self.concept_type_single:
            t_maximum_rollout_depth = 20 # This is the double of 'both' concept type due to the fact that the opponent's turn is skipped
            maximum_depth_find_sub_rollout = t_maximum_rollout_depth - 10
        else:
            t_maximum_rollout_depth = 10 # Paper suggests 10 or 5
            maximum_depth_find_sub_rollout = t_maximum_rollout_depth - 5

        optimal_rollout_states = []
        subpar_rollout_states = []
        
        # Starting from the root node
        node = self.mcts.root

        optimal_rollout_states.append(node.predict_state_rep)

        if self.concept_type_single:
            # Skip the oposing players turn by choosing the best action
            highest_visit_count = -1
            # Find the best action for the current player (opponent playing optimally)
            for child in node.children:
                if child.n_visit_count > highest_visit_count:
                    highest_visit_count = child.n_visit_count
                    node = child

        while node and node.time_step < t_maximum_rollout_depth:
            # Find the optimal next state given visit count
            highest_visit_count = -1
            next_optimal_node = None

            for child in node.children:
                if child.n_visit_count > highest_visit_count:
                    highest_visit_count = child.n_visit_count
                    next_optimal_node = child
            
            if next_optimal_node is None:
                break

            if next_optimal_node.predict_state_rep is not None:
                optimal_rollout_states.append(next_optimal_node.predict_state_rep)

            # From the subpar node, perform a optimal rollout to the maximum depth
            if node.time_step < maximum_depth_find_sub_rollout:
                # Find the subpar next state with a minimum value difference of 0.20 and/or a visit count difference of 10% of the highest visit count
                sub_par_children: list[Node] = []
                for child in node.children:
                    if child.n_visit_count < highest_visit_count * (1-min_visit_count_diff) or child.q_value() < next_optimal_node.q_value() - min_value_diff:
                        sub_par_children.append(child)
                
                # Find the best subpar state to rollout
                best_subpar_node = None
                highest_sub_par_visit_count = -1

                for child in sub_par_children:
                    if child.n_visit_count > highest_sub_par_visit_count:
                        highest_sub_par_visit_count = child.n_visit_count
                        best_subpar_node = child

                # Assert that the best subpar state is has a lower visit count than the optimal state
                assert best_subpar_node.n_visit_count < next_optimal_node.n_visit_count

                if best_subpar_node.predict_state_rep is not None:
                    subpar_rollout_states.append(best_subpar_node.predict_state_rep)
                
                curr_node = best_subpar_node
                moves = 0
                for _ in range(t_maximum_rollout_depth - node.time_step):
                    highest_visit_count = -1
                    optimal_child = None
                    for child in curr_node.children:
                        if child.n_visit_count > highest_visit_count:
                            highest_visit_count = child.n_visit_count
                            optimal_child = child
                    
                    if optimal_child:
                        if optimal_child.predict_state_rep is not None:
                            if self.concept_type_single:
                                # Only add the state if the current player is playing (opponent is skipped)
                                if moves % 2 != 0:
                                    subpar_rollout_states.append(optimal_child.predict_state_rep)
                            else:
                                subpar_rollout_states.append(optimal_child.predict_state_rep)
                            curr_node = optimal_child
                        else:
                            break
                    
                    moves += 1
            
            # If the concept is single, skip the oposing players turn by choosing the best action
            # If the concept is both, choose the best state from current player and state
            if self.concept_type_single:
                # Skip the oposing players turn by choosing the best action
                curr_node = next_optimal_node
                node = None
                highest_visit_count = -1
                # Find the best action for the current player (opponent playing optimally)
                for child in curr_node.children:
                    if child.n_visit_count > highest_visit_count:
                        highest_visit_count = child.n_visit_count
                        node = child
            else:
                # Choose the best state for the current player
                node = next_optimal_node
        
        return optimal_rollout_states, subpar_rollout_states
    
    """
    Concept function for different initialization states
    """
    
    @staticmethod
    def opening_play(board_size, komi=0.5):
        """
        Concept of starting from an empty board
        """
        go_env = env.GoEnv(board_size, komi)
        go_env.reset()
        return go_env.canonical_state()
