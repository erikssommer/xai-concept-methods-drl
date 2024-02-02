import unittest
import numpy as np

import os
# Add the src folder to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from env import GoEnv
from env import govars
from mcts import MCTS
from policy import ConvNet, ResNet, FastPredictor, LiteModel


class TestMCTSvsRandom(unittest.TestCase):

    def test_mcts_as_black_vs_random(self):
        victories = 0
        games = 10
        simulations = 100
        board_size = 5
        move_cap = board_size**2 * 4
        komi = 0.5
        c = 1.3
        det_moves = 0
        path = '../models/training/board_size_5/net_10000.keras'
        policy = ConvNet(board_size, path)

        model = FastPredictor(LiteModel.from_keras_model(policy.model))

        for _ in range(games):
            go_env = GoEnv(size=board_size, komi=komi)
            go_env.reset()
            game_state = go_env.canonical_state()
            tree = MCTS(game_state, simulations, board_size, move_cap, model, c, komi, det_moves)

            curr_turn = 0
            move_nr = 0

            terminated = False

            while not terminated:
                if curr_turn == govars.BLACK:
                    best_action_node, _ = tree.search(move_nr)
                    _, _, terminated, _ = go_env.step(best_action_node.action)
                    tree.set_root_node(best_action_node)
                else:
                    action = go_env.uniform_random_action()
                    _, _, terminated, _ = go_env.step(action)
                    
                    # Update the tree, may be done better
                    tree.set_root_node_with_action(action)
                
                curr_turn = 1 - curr_turn
                move_nr += 1
                go_env.render()

            black_won = go_env.winning()

            if black_won == 1:
                victories += 1

        # Calculate the win probability
        win_probability = victories / games

        # Print the results
        print("Win probability as black: {}".format(win_probability))

        # Assert the results
        self.assertTrue(win_probability >= 0.9)

    
    def test_mcts_as_white_with_only_critic_vs_random(self):
        victories = 0
        games = 10
        simulations = 100
        board_size = 5
        move_cap = board_size**2 * 4
        komi = 0.5
        c = 1.3
        det_moves = 0
        path = '../models/training/board_size_5/net_10000.keras'
        policy = ConvNet(board_size, path)

        model = FastPredictor(LiteModel.from_keras_model(policy.model))

        for _ in range(games):
            go_env = GoEnv(size=board_size)
            go_env.reset()
            game_state = go_env.canonical_state()
            tree = MCTS(game_state, simulations, board_size, move_cap, model, c, komi, det_moves)

            curr_turn = 0
            move_nr = 0
            
            first_action = True
            terminated = False

            while not terminated:
                if curr_turn == govars.WHITE:
                    best_action_node, _ = tree.search(move_nr)
                    _, _, terminated, _ = go_env.step(best_action_node.action)
                    tree.set_root_node(best_action_node)
                else:
                    action = go_env.uniform_random_action()
                    state, _, terminated, _ = go_env.step(action)
                    
                    # Need to set the root node to the state if it is the first move of the game
                    if first_action:
                        tree.set_root(state, 1)
                        first_action = False
                    else:
                        tree.set_root_node_with_action(action)
                
                curr_turn = 1 - curr_turn
                move_nr += 1
                go_env.render()

            black_won = go_env.winning()

            if black_won == -1:
                victories += 1

        # Calculate the win probability
        win_probability = victories / games

        # Print the results
        print("Win probability as white: {}".format(win_probability))

        # Assert the results
        self.assertTrue(win_probability >= 0.9)




if __name__ == '__main__':
    unittest.main()
