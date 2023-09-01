import unittest
import gym

import sys
sys.path.insert(0, 'src')

from mcts.mcts import MCTS


class TestMCTSvsRandom(unittest.TestCase):
    def test_mcts_vs_random(self):
        victories = 0
        games = 1
        rollouts = 100
        mcts_player = 0

        for _ in range(games):
            go_env = gym.make('gym_go:go-v0', size=4)
            go_env.reset()
            game_state = go_env.canonical_state()
            curr_player = go_env.turn()
            print(curr_player)
            tree = MCTS(1, 1, rollouts, 1.3)
            tree.set_root(game_state)

            terminated = False

            while not terminated:
                if curr_player == mcts_player:
                    print("MCTS")
                    best_action_node, player, game_state, distribution = tree.search(curr_player)
                    observation, reward, terminated, info = go_env.step(best_action_node.action)
                    tree.root = best_action_node
                else:
                    print("Random")
                    valid_moves = go_env.valid_moves()
                    valid = False

                    while not valid:
                        action = go_env.action_space.sample()
                        if valid_moves[action] != 0:
                            observation, reward, terminated, info = go_env.step(action)
                            valid = True
                    
                    # Update the tree
                    tree = MCTS(1, 1, rollouts, 1.3)
                    tree.set_root(observation)
                
                curr_player = go_env.turn()

            if go_env.winner() == mcts_player:
                victories += 1

        # Calculate the win probability
        win_probability = victories / games

        # Print the results
        print("Win probability: {}".format(win_probability))

        # Assert the results
        self.assertTrue(win_probability > 0.9)


if __name__ == '__main__':
    unittest.main()
