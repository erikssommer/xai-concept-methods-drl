from mcts import MCTS
from mcts import Node
from policy import ConvNet, ResNet, FastPredictor, LiteModel
from typing import List, Tuple
import numpy as np

"""
Dynamic concepts are concepts that are not fixed, but rather change over time in response to the environment.
"""


class DynamicConcepts:
    def __init__(self,
                 init_state,
                 simulations,
                 board_size,
                 concept_type_single,
                 path,
                 random_subpar=False,
                 resnet=False,
                 move_cap=100):
        # 'both' or 'single'.
        # Single means concepts where only one player is considered
        # Both means concepts where both players are considered.
        self.concept_type_single = concept_type_single
        self.board_size = board_size
        self.random_subpar = random_subpar
        
        # Subpar variations
        self.min_visit_count_diff = 0.1
        self.min_value_diff = 0.2

        if self.concept_type_single:
            # This is the double of 'both' concept type due to the fact that the opponent's turn is skipped.
            self.t_maximum_rollout_depth = 20
            self.maximum_depth_find_sub_rollout = self.t_maximum_rollout_depth - 10
        else:
            self.t_maximum_rollout_depth = 10  # Paper suggests 10 or 5
            self.maximum_depth_find_sub_rollout = self.t_maximum_rollout_depth - 5

        if resnet:
            neural_network = ResNet(board_size, load_path=path)
        else:
            neural_network = ConvNet(board_size, load_path=path)

        predictor = FastPredictor(
            LiteModel.from_keras_model(neural_network.model))

        # Initialize the MCTS
        self.mcts = MCTS(init_state, simulations,
                         board_size, move_cap, predictor)
        

    def generate_cases(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Create the tree
        self.mcts.reset_root()
        self.mcts.search()
        self.add_all_stete_reps()

        optimal_rollout_states: List[np.ndarray] = []
        subpar_rollout_states: List[np.ndarray] = []

        # Starting from the root node
        node = self.mcts.root

        optimal_rollout_states.append(node.predict_state_rep)
        self.mcts.root.optiaml_rollout = True

        if self.concept_type_single:
            # Skip the opposing players turn by choosing the best action
            node, _ = self.find_best_next_node(node.children)

        while node.time_step < self.t_maximum_rollout_depth:
            # Find the optimal next state given visit count
            next_optimal_node, highest_visit_count = self.find_best_next_node(node.children)

            if next_optimal_node is None:
                break

            optimal_rollout_states.append(next_optimal_node.predict_state_rep)
            next_optimal_node.optiaml_rollout = True

            # From the subpar node, perform a optimal rollout to the maximum depth
            if node.time_step < self.maximum_depth_find_sub_rollout:
                # Find the subpar next state with a minimum value difference of 0.20 and/or
                # a visit count difference of 10% of the highest visit count
                sub_par_children: list[Node] = []
                for child in node.children:
                    if child.n_visit_count < highest_visit_count * (1 - self.min_visit_count_diff) \
                            or child.q_value() < next_optimal_node.q_value() - self.min_value_diff:
                        sub_par_children.append(child)

                best_subpar_node, _ = self.find_best_next_node(sub_par_children)

                if best_subpar_node is None:
                    break

                # Assert that the best subpar state is has a lower visit count than the optimal state
                assert best_subpar_node.n_visit_count < next_optimal_node.n_visit_count \
                    or best_subpar_node.q_value() < next_optimal_node.q_value(), "Subpar state is not subpar enough!"

                subpar_rollout_states.append(best_subpar_node.predict_state_rep)
                best_subpar_node.optiaml_rollout = False

                curr_node = best_subpar_node

                for moves in range(self.t_maximum_rollout_depth - node.time_step):
                    if len(curr_node.children) == 0:
                        break
                    if self.random_subpar:
                        # Loop through all children and collect the state representations
                        optimal_child = self.find_best_subpar_node(curr_node.children)
                        #optimal_child = np.random.choice(curr_node.children) # Testing
                    else:
                        optimal_child, _ = self.find_best_next_node(curr_node.children)

                    if optimal_child is None:
                        break

                    if self.concept_type_single and moves % 2 != 0:
                        # Only add the state if the current player is playing (opponent is skipped)
                        subpar_rollout_states.append(optimal_child.predict_state_rep)
                        optimal_child.optiaml_rollout = False
                    else:
                        subpar_rollout_states.append(optimal_child.predict_state_rep)
                        optimal_child.optiaml_rollout = False
                    curr_node = optimal_child 

            # If the concept is single, skip the oposing players turn by choosing the best action
            # If the concept is both, choose the best state from current player and state
            if self.concept_type_single:
                # Skip the oposing players turn by choosing the best action
                node, _ = self.find_best_next_node(next_optimal_node.children)
            else:
                # Choose the best state for the current player
                node = next_optimal_node

        return optimal_rollout_states, subpar_rollout_states

    def find_best_next_node(self, children: List[Node]) -> Tuple[Node, int]:
        highest_visit_count = -1
        next_node = None
        for child in children:
            if child.n_visit_count > highest_visit_count:
                highest_visit_count = child.n_visit_count
                next_node = child

        return next_node, highest_visit_count
    
    def find_best_subpar_node(self, children: List[Node]) -> Node:
        # Find the best subpar node
        next_optimal_node, highest_visit_count = self.find_best_next_node(children)

        sub_par_children: list[Node] = []
        for child in children:
            if child.n_visit_count < highest_visit_count * (1 - self.min_visit_count_diff) \
                        or child.q_value() < next_optimal_node.q_value() - self.min_value_diff:
                sub_par_children.append(child)

        best_subpar_node, _ = self.find_best_next_node(sub_par_children)

        return best_subpar_node
    
    def add_all_stete_reps(self):
        # Create a stack to perform post-order traversal
        stack = [self.mcts.root]
        while stack:
            node = stack.pop()
            
            # Add the state representation to the node
            if node.predict_state_rep is None:
                node.model_state_format(self.board_size)
            
            # Add the children of the current node to the stack
            stack.extend(node.children)


    def view_tree(self):
        return self.mcts.view_tree()

    """
    Concept functions for different initialization states
    """

    @staticmethod
    def opening_play(board_size: int) -> Tuple[np.ndarray, bool, str]:
        """
        This is the initial phase of the game where players try to establish their initial structures and influence on the board. 
        The goal is to control as much territory as possible and set up for the middle game.
        """
        concept_type_single = False
        game_state = np.zeros((6, board_size, board_size))

        return game_state, concept_type_single, "opening_play"

    @staticmethod
    def end_game(board_size: int) -> Tuple[np.ndarray, bool, str]:
        """
        The end game is the final phase of the game where players try to secure their territories and capture their opponent's stones.
        """
        concept_type_single = False
        # Initialize all planes to zeros
        game_state = np.zeros((6, board_size, board_size))

        # Set up an endgame scenario
        for i in range(board_size):
            for j in range(board_size):
                if i < board_size / 2 and j < board_size / 2:
                    game_state[0, i, j] = 1  # Current player's territory
                elif i >= board_size / 2 and j >= board_size / 2:
                    game_state[1, i, j] = 1  # Opponent player's territory

        # Set the third plane to represent the current player's turn
        game_state[2, :, :] = 0

        # Set the fourth plane to represent invalid moves (places where there are stones)
        # Loop through the board and set the fourth plane to 1 where there are stones
        for i in range(board_size):
            for j in range(board_size):
                if game_state[0, i, j] == 1 or game_state[1, i, j] == 1:
                    game_state[3, i, j] = 1
        
        # Set the fifth plane to represent that the previous move was not a pass
        game_state[4, :, :] = 0

        # Set the last plane to represent that the game is not over
        game_state[5, :, :] = 0

        return game_state, concept_type_single, "end_game"

    @staticmethod
    def life_and_death(board_size: int) -> Tuple[np.ndarray, bool, str]:
        """
        Concept of starting from a board with a few stones
        """
        concept_type_single = False
        # Initialize all planes to zeros
        game_state = np.zeros((6, board_size, board_size))

        # Set up a scenario where Player 1 has a group of stones under threat
        game_state[0, board_size // 2 +1, board_size // 2 +1] = 1  # Current player's stone
        game_state[1, board_size // 2, board_size // 2 + 1] = 1  # Opponent player's stone
        game_state[1, board_size // 2 + 1, board_size // 2] = 1  # Opponent player's stone
        game_state[1, board_size // 2 - 1, board_size // 2] = 1  # Opponent player's stone
        game_state[1, board_size // 2, board_size // 2 - 1] = 1  # Opponent player's stone

        # Set the third plane to represent the current player's turn
        game_state[2, :, :] = 1

        # Set the fourth plane to represent invalid moves (all territories are claimed)
        for i in range(board_size):
            for j in range(board_size):
                if game_state[0, i, j] == 1 or game_state[1, i, j] == 1:
                    game_state[3, i, j] = 1
        
        # If ko is present, set a ko stone
        game_state[3, board_size // 2, board_size // 2] = 1  # Current player's stone

        # Set the fifth plane to represent that the previous move was not a pass
        game_state[4, :, :] = 0

        # Set the last plane to represent that the game is not over
        game_state[5, :, :] = 0

        return game_state, concept_type_single, "life_and_death"


    @staticmethod
    def keep_initiative(board_size: int) -> Tuple[np.ndarray, bool, str]:
        """
        Concept of starting from a board with a few stones
        """
        concept_type_single = False
        # Initialize all planes to zeros
        game_state = np.zeros((6, board_size, board_size))

        # Set up a scenario where Player 1 has the initiative
        game_state[0, board_size // 2, board_size // 2] = 1  # Current player's stone
        game_state[0, board_size // 2, board_size // 2+1] = 1  # Current player's stone
        game_state[0, board_size // 2, board_size // 2-1] = 1  # Current player's stone
        game_state[1, board_size // 2+1, board_size // 2] = 1  # Opponent player's stone
        game_state[1, board_size // 2+1, board_size // 2-1] = 1  # Opponent player's stone
        game_state[1, board_size // 2+1, board_size // 2+1] = 1  # Opponent player's stone

        # Set the third plane to represent the current player's turn
        game_state[2, :, :] = 1

        # Set the fourth plane to represent invalid moves (all territories are claimed)
        for i in range(board_size):
            for j in range(board_size):
                if game_state[0, i, j] == 1 or game_state[1, i, j] == 1:
                    game_state[3, i, j] = 1

        # Set the fifth plane to represent that the previous move was not a pass
        game_state[4, :, :] = 0

        # Set the last plane to represent that the game is not over
        game_state[5, :, :] = 0

        return game_state, concept_type_single, "keep_initiative"

    @staticmethod
    def ko_fight(board_size: int) -> Tuple[np.ndarray, bool, str]:
        """
        A Ko fight involves a sequence of moves elsewhere on the board (Ko threats) that aim to make the opponent respond so that the player can retake the Ko.
        """
        concept_type_single = False
        # Initialize all planes to zeros
        game_state = np.zeros((6, board_size, board_size))

        # Set up a scenario where Player 1 has a ko to fight for
        game_state[0, board_size // 2-1, board_size // 2] = 1
        game_state[0, board_size // 2+1, board_size // 2] = 1
        game_state[0, board_size // 2, board_size // 2+1] = 1
        game_state[0, board_size // 2, board_size // 2+2] = 1
        game_state[1, board_size // 2, board_size // 2] = 1
        game_state[1, board_size // 2-1, board_size // 2-1] = 1
        game_state[1, board_size // 2+1, board_size // 2-1] = 1
        game_state[1, board_size // 2, board_size // 2-2] = 1

        # Set the third plane to represent the current player's turn
        game_state[2, :, :] = 0

        # Set the fourth plane to represent invalid moves (all territories are claimed)
        for i in range(board_size):
            for j in range(board_size):
                if game_state[0, i, j] == 1 or game_state[1, i, j] == 1:
                    game_state[3, i, j] = 1

        # Set the fifth plane to represent that the previous move was not a pass
        game_state[4, :, :] = 0

        # Set the last plane to represent that the game is not over
        game_state[5, :, :] = 0
        
        return game_state, concept_type_single, "ko_fight"
        


    @staticmethod
    def invasion_and_reduction(board_size: int) -> Tuple[np.ndarray, bool, str]:
        """
        These are strategies used to disrupt your opponent's territories. 
        An invasion is a sequence of moves that attempts to establish a live group inside an opponent's territory, 
        while a reduction is a move or sequence of moves that attempts to reduce the potential size
        of an opponent's territory without necessarily trying to live inside.
        """
        concept_type_single = False
        # Initialize all planes to zeros
        game_state = np.zeros((6, board_size, board_size))

        # Set up a scenario where Player 1 has the initiative
        game_state[0, board_size // 2, board_size // 2] = 1  # Current player's stone
        game_state[0, board_size // 2, board_size // 2+1] = 1  # Current player's stone
        game_state[0, board_size // 2, board_size // 2-1] = 1  # Current player's stone
        game_state[0, board_size // 2, board_size // 2+2] = 1  # Current player's stone
        game_state[0, board_size // 2-1, board_size // 2-2] = 1  # Current player's stone
        game_state[1, board_size // 2+1, board_size // 2] = 1  # Opponent player's stone
        game_state[1, board_size // 2+1, board_size // 2-1] = 1  # Opponent player's stone
        game_state[1, board_size // 2+1, board_size // 2+1] = 1  # Opponent player's stone

        # White invades black
        game_state[1, board_size // 2-1, board_size // 2+1] = 1

        # Set the third plane to represent the current player's turn
        game_state[2, :, :] = 1

        # Set the fourth plane to represent invalid moves (all territories are claimed)
        for i in range(board_size):
            for j in range(board_size):
                if game_state[0, i, j] == 1 or game_state[1, i, j] == 1:
                    game_state[3, i, j] = 1

        # Set the fifth plane to represent that the previous move was not a pass
        game_state[4, :, :] = 0

        # Set the last plane to represent that the game is not over
        game_state[5, :, :] = 0
        
        return game_state, concept_type_single, "invasion_and_reduction"
