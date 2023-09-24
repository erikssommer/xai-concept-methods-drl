import unittest
import gym

import sys
sys.path.insert(0, 'ml')

from ml.mcts.mcts import MCTS
from ml.game.data import GoVars

# Set the logging level
gym.logger.set_level(40)


class TestMCTSvsRandom(unittest.TestCase):
    def test_mcts_as_black_vs_random(self):
        victories = 0
        games = 5
        rollouts = 50
        board_size = 5

        for _ in range(games):
            go_env = gym.make('gym_go:go-v0', size=board_size)
            go_env.reset()
            game_state = go_env.canonical_state()
            curr_turn = go_env.turn()
            tree = MCTS(game_state, 1, 1, rollouts, board_size**2, board_size**2*5)
            tree.set_root(game_state)

            terminated = False

            while not terminated:
                if curr_turn == GoVars.BLACK:
                    best_action_node, game_state, distribution = tree.search()
                    state, reward, terminated, info = go_env.step(best_action_node.action)
                else:
                    action = go_env.uniform_random_action()
                    state, reward, terminated, info = go_env.step(action)
                    
                    # Update the tree, may be done better
                    tree = MCTS(1, 1, rollouts)
                    tree.set_root(state)
                
                curr_turn = go_env.turn()
                go_env.render()

            black_won = go_env.winning()

            if black_won == 1:
                victories += 1
            
            go_env.render()

        # Calculate the win probability
        win_probability = victories / games

        # Print the results
        print("Win probability: {}".format(win_probability))

        # Assert the results
        self.assertTrue(win_probability >= 0.9)
    
    def test_mcts_as_white_vs_random(self):
        victories = 0
        games = 5
        rollouts = 300
        board_size = 5

        for _ in range(games):
            go_env = gym.make('gym_go:go-v0', size=board_size)
            go_env.reset()
            game_state = go_env.canonical_state()
            curr_turn = go_env.turn()
            tree = MCTS(1, 1, rollouts)
            tree.set_root(game_state)

            terminated = False

            while not terminated:
                if curr_turn == GoVars.WHITE:
                    best_action_node, game_state, distribution = tree.search()
                    state, reward, terminated, info = go_env.step(best_action_node.action)
                else:
                    action = go_env.uniform_random_action()
                    state, reward, terminated, info = go_env.step(action)
                    
                    # Update the tree, may be done better
                    tree = MCTS(1, 1, rollouts)
                    tree.set_root(state)
                
                curr_turn = go_env.turn()
                go_env.render()

            black_won = go_env.winning()

            if black_won != 1:
                victories += 1
            
            go_env.render()

        # Calculate the win probability
        win_probability = victories / games

        # Print the results
        print("Win probability: {}".format(win_probability))

        # Assert the results
        self.assertTrue(win_probability >= 0.9)


if __name__ == '__main__':
    unittest.main()
