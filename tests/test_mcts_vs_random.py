import unittest

import os
# Add the src folder to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from env import GoEnv
from env import govars
from mcts import MCTS


class TestMCTSvsRandom(unittest.TestCase):
    def test_mcts_as_black_vs_random(self):
        victories = 0
        games = 5
        rollouts = 50
        board_size = 4

        for _ in range(games):
            go_env = GoEnv(size=board_size)
            go_env.reset()
            game_state = go_env.state()
            curr_turn = go_env.turn()
            tree = MCTS(game_state, 1, 1, rollouts, board_size, board_size**2*5)

            terminated = False

            while not terminated:
                if curr_turn == govars.BLACK:
                    best_action_node, game_state, _ = tree.search()
                    _, _, terminated, _ = go_env.step(best_action_node.action)
                    tree.set_root_node(best_action_node)
                else:
                    action = go_env.uniform_random_action()
                    _, _, terminated, _ = go_env.step(action)
                    
                    # Update the tree, may be done better
                    tree.set_root_node_with_action(action)
                
                curr_turn = go_env.turn()
                #go_env.render()

            black_won = go_env.winning()

            if black_won == 1:
                victories += 1
            
            #go_env.render()

        # Calculate the win probability
        win_probability = victories / games

        # Print the results
        print("Win probability as black: {}".format(win_probability))

        # Assert the results
        self.assertTrue(win_probability >= 0.9)
    
    def test_mcts_as_white_vs_random(self):
        victories = 0
        games = 5
        rollouts = 50
        board_size = 4

        for _ in range(games):
            go_env = GoEnv(size=board_size)
            go_env.reset()
            game_state = go_env.state()
            curr_turn = go_env.turn()
            tree = MCTS(game_state, 1, 1, rollouts, board_size, board_size**2*5)

            first_action = True
            terminated = False

            while not terminated:
                if curr_turn == govars.WHITE:
                    best_action_node, game_state, _ = tree.search()
                    _, _, terminated, _ = go_env.step(best_action_node.action)

                    tree.set_root_node(best_action_node)
                else:
                    action = go_env.uniform_random_action()
                    state, _, terminated, _ = go_env.step(action)

                    # Need to set the root node to the state if it is the first move of the game
                    if first_action:
                        tree.set_root(state)
                        first_action = False
                    else:
                        tree.set_root_node_with_action(action)
                
                curr_turn = go_env.turn()
                #go_env.render()

            black_won = go_env.winning()

            if black_won != 1:
                victories += 1
            
            #go_env.render()

        # Calculate the win probability
        win_probability = victories / games

        # Print the results
        print("Win probability as white: {}".format(win_probability))

        # Assert the results
        self.assertTrue(win_probability >= 0.9)


if __name__ == '__main__':
    unittest.main()
