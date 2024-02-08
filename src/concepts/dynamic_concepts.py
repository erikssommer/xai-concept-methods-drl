from mcts import MCTS
from mcts import Node
from policy import ConvNet, FastPredictor, LiteModel
import env
from typing import List, Tuple

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
        self.mcts.root.optiaml_rollout = True

        if self.concept_type_single:
            # Skip the opposing players turn by choosing the best action
            node, _ = self.find_best_next_node(node.children)

        while node and node.time_step < t_maximum_rollout_depth:
            # Find the optimal next state given visit count
            next_optimal_node, highest_visit_count = self.find_best_next_node(node.children)
            
            if next_optimal_node is None:
                break

            if next_optimal_node.predict_state_rep is not None:
                optimal_rollout_states.append(next_optimal_node.predict_state_rep)
                next_optimal_node.optiaml_rollout = True

            # From the subpar node, perform a optimal rollout to the maximum depth
            if node.time_step < maximum_depth_find_sub_rollout:
                # Find the subpar next state with a minimum value difference of 0.20 and/or a visit count difference of 10% of the highest visit count
                sub_par_children: list[Node] = []
                for child in node.children:
                    if child.n_visit_count < highest_visit_count * (1-min_visit_count_diff) or child.q_value() < next_optimal_node.q_value() - min_value_diff:
                        sub_par_children.append(child)
                
                # Find the best subpar state to rollout
                if len(sub_par_children) == 0:
                    break

                best_subpar_node, _ = self.find_best_next_node(sub_par_children)

                # Assert that the best subpar state is has a lower visit count than the optimal state
                assert best_subpar_node.n_visit_count < next_optimal_node.n_visit_count or best_subpar_node.q_value() < next_optimal_node.q_value()

                if best_subpar_node.predict_state_rep is not None:
                    subpar_rollout_states.append(best_subpar_node.predict_state_rep)
                    best_subpar_node.optiaml_rollout = False
                
                curr_node = best_subpar_node
                moves = 0
                for _ in range(t_maximum_rollout_depth - node.time_step):
                    optimal_child, _ = self.find_best_next_node(curr_node.children)
                    
                    if optimal_child:
                        if optimal_child.predict_state_rep is not None:
                            if self.concept_type_single:
                                # Only add the state if the current player is playing (opponent is skipped)
                                if moves % 2 != 0:
                                    subpar_rollout_states.append(optimal_child.predict_state_rep)
                                    optimal_child.optiaml_rollout = False
                            else:
                                subpar_rollout_states.append(optimal_child.predict_state_rep)
                                optimal_child.optiaml_rollout = False
                            curr_node = optimal_child
                        else:
                            break
                    
                    moves += 1
            
            # If the concept is single, skip the oposing players turn by choosing the best action
            # If the concept is both, choose the best state from current player and state
            if self.concept_type_single:
                # Skip the oposing players turn by choosing the best action
                node, _ = self.find_best_next_node(next_optimal_node.children)
            else:
                # Choose the best state for the current player
                node = next_optimal_node
        
        return optimal_rollout_states, subpar_rollout_states
    
    """
    Concept function for different initialization states
    """

    def find_best_next_node(self, children: List[Node]) -> Tuple[Node, int]:
        highest_visit_count = -1
        next_node = None
        for child in children:
            if child.n_visit_count > highest_visit_count:
                highest_visit_count = child.n_visit_count
                next_node = child
        
        return next_node, highest_visit_count
    
    def view_tree(self):
        return self.mcts.view_tree()
    
    @staticmethod
    def opening_play(board_size, komi=0.5):
        """
        Concept of starting from an empty board
        """
        concept_type_single = False
        go_env = env.GoEnv(board_size, komi)
        go_env.reset()
        return go_env.canonical_state(), concept_type_single
