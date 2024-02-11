from mcts import MCTS
from mcts import Node
from policy import ConvNet, FastPredictor, LiteModel
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
                 move_cap=100,
                 random_subpar=True):
        # 'both' or 'single'.
        # Single means concepts where only one player is considered
        # Both means concepts where both players are considered.
        self.concept_type_single = concept_type_single
        self.board_size = board_size
        self.random_subpar = random_subpar

        neural_network = ConvNet(board_size, load_path=path)

        predictor = FastPredictor(
            LiteModel.from_keras_model(neural_network.model))

        # Initialize the MCTS
        self.mcts = MCTS(init_state, simulations,
                         board_size, move_cap, predictor)

    def generate_cases(self):
        # Create the tree
        self.mcts.search()

        # Subpar variations
        min_visit_count_diff = 0.1
        min_value_diff = 0.2

        if self.concept_type_single:
            # This is the double of 'both' concept type due to the fact that the opponent's turn is skipped.
            t_maximum_rollout_depth = 20
            maximum_depth_find_sub_rollout = t_maximum_rollout_depth - 10
        else:
            t_maximum_rollout_depth = 10  # Paper suggests 10 or 5
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
            next_optimal_node, highest_visit_count = self.find_best_next_node(
                node.children)

            if next_optimal_node is None:
                break

            if next_optimal_node.predict_state_rep is not None:
                optimal_rollout_states.append(
                    next_optimal_node.predict_state_rep)
                next_optimal_node.optiaml_rollout = True

            # From the subpar node, perform a optimal rollout to the maximum depth
            if node.time_step < maximum_depth_find_sub_rollout:
                # Find the subpar next state with a minimum value difference of 0.20 and/or
                # a visit count difference of 10% of the highest visit count
                sub_par_children: list[Node] = []
                for child in node.children:
                    if child.n_visit_count < highest_visit_count * (1 - min_visit_count_diff) \
                            or child.q_value() < next_optimal_node.q_value() - min_value_diff:
                        sub_par_children.append(child)

                # If the subpar children list is empty, break the loop
                if len(sub_par_children) == 0:
                    break

                best_subpar_node, _ = self.find_best_next_node(
                    sub_par_children)

                # Assert that the best subpar state is has a lower visit count than the optimal state
                assert best_subpar_node.n_visit_count < next_optimal_node.n_visit_count \
                    or best_subpar_node.q_value() < next_optimal_node.q_value()

                if best_subpar_node.predict_state_rep is not None:
                    subpar_rollout_states.append(
                        best_subpar_node.predict_state_rep)
                    best_subpar_node.optiaml_rollout = False

                curr_node = best_subpar_node
                moves = 0
                for _ in range(t_maximum_rollout_depth - node.time_step):
                    if self.random_subpar:
                        optimal_child = np.random.choice(curr_node.children)
                    else:
                        optimal_child, _ = self.find_best_next_node(
                            curr_node.children)

                    if optimal_child:
                        if optimal_child.predict_state_rep is not None:
                            if self.concept_type_single:
                                # Only add the state if the current player is playing (opponent is skipped)
                                if moves % 2 != 0:
                                    subpar_rollout_states.append(
                                        optimal_child.predict_state_rep)
                                    optimal_child.optiaml_rollout = False
                            else:
                                subpar_rollout_states.append(
                                    optimal_child.predict_state_rep)
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

    """
    Concept functions for different initialization states
    """

    @staticmethod
    def opening_play(board_size):
        """
        This is the initial phase of the game where players try to establish their initial structures and influence on the board. 
        The goal is to control as much territory as possible and set up for the middle game.
        """
        concept_type_single = False
        game_state = np.zeros((6, board_size, board_size))

        return game_state, concept_type_single

    @staticmethod
    def end_game(board_size):
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

        return game_state, concept_type_single

    @staticmethod
    def life_and_death(board_size):
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

        print(game_state)

        return game_state, concept_type_single


    @staticmethod
    def keep_initiative(board_size):
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

        return game_state, concept_type_single

    @staticmethod
    def ko_fight(board_size):
        """
        A Ko fight involves a sequence of moves elsewhere on the board (Ko threats) that aim to make the opponent respond so that the player can retake the Ko.
        """
        concept_type_single = False
        # Initialize all planes to zeros
        game_state = np.zeros((6, board_size, board_size))

        # Set up a scenario where Player 1 has a ko to fight for
        game_state[0, board_size // 2, board_size // 2] = 1
        game_state[0, board_size // 2-1, board_size // 2-1] = 1
        game_state[0, board_size // 2+1, board_size // 2-1] = 1
        game_state[0, board_size // 2, board_size // 2-2] = 1
        game_state[1, board_size // 2-1, board_size // 2] = 1
        game_state[1, board_size // 2+1, board_size // 2] = 1
        game_state[1, board_size // 2, board_size // 2+1] = 1

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
        
        return game_state, concept_type_single
        


    @staticmethod
    def invasion_and_reduction(board_size):
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
        
        return game_state, concept_type_single
