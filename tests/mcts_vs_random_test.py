import unittest
import gym

import sys
sys.path.insert(0, 'src')

from mcts.mcts import MCTS
from game.data import GoGame
from game.data import GoVars


class TestMCTSvsRandom(unittest.TestCase):
    def test_mcts_vs_random(self):
        victories = 0
        games = 5
        rollouts = 100
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
                if curr_turn == GoVars.BLACK:
                    best_action_node, player, game_state, distribution = tree.search(curr_turn)
                    observation, reward, terminated, info = go_env.step(best_action_node.action)
                    tree.root = best_action_node
                else:
                    action = go_env.uniform_random_action()
                    state, reward, terminated, info = go_env.step(action)
                    
                    # Update the tree
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


if __name__ == '__main__':
    unittest.main()
