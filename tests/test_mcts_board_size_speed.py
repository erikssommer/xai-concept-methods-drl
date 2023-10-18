import unittest

import os
# Add the src folder to the path
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from mcts import MCTS
import env
from policy import ActorCriticNet

class TestMctsBoardSizeSpeed(unittest.TestCase):
    def test_without_model(self):
        board_size = 5 # 5x5
        ROLLOUTS = 100


        go_env = env.GoEnv(size=board_size)
        game_state = go_env.state()
        tree = MCTS(game_state, 1, 1, ROLLOUTS, board_size, board_size**2*5)

        # Take the time it takes to run rollouts
        start = time.time()
        tree.search()
        end = time.time()
        time_taken = end - start
        # Make it 3 decimal places
        time_taken = round(time_taken, 3)
        print(f"Time taken for {ROLLOUTS} rollouts on a {board_size}x{board_size} board: {time_taken} seconds")

        board_size = 8
        
        go_env = env.GoEnv(size=board_size)
        game_state = go_env.state()
        tree = MCTS(game_state, 1, 1, ROLLOUTS, board_size, board_size**2*5)

        # Take the time it takes to run rollouts
        start = time.time()
        tree.search()
        end = time.time()
        time_taken = end - start
        # Make it 3 decimal places
        time_taken = round(time_taken, 3)
        print(f"Time taken for {ROLLOUTS} rollouts on a {board_size}x{board_size} board: {time_taken} seconds")

    def test_with_model(self):
        board_size = 5
        ROLLOUTS = 1000

        go_env = env.GoEnv(size=board_size)
        game_state = go_env.state()
        tree = MCTS(game_state, 1, 0, ROLLOUTS, board_size, board_size**2*5, policy_nn=ActorCriticNet(board_size, summary=False))

        # Take the time it takes to run rollouts
        start = time.time()
        tree.search()
        end = time.time()
        time_taken = end - start
        # Make it 3 decimal places
        time_taken = round(time_taken, 3)

        print(f"Time taken for {ROLLOUTS} rollouts on a {board_size}x{board_size} board with model: {time_taken} seconds")

        board_size = 8

        go_env = env.GoEnv(size=board_size)
        game_state = go_env.state()
        tree = MCTS(game_state, 1, 0, ROLLOUTS, board_size, board_size**2*5, policy_nn=ActorCriticNet(board_size, summary=False))

        # Take the time it takes to run rollouts
        start = time.time()
        tree.search()
        end = time.time()
        time_taken = end - start
        # Make it 3 decimal places
        time_taken = round(time_taken, 3)

        print(f"Time taken for {ROLLOUTS} rollouts on a {board_size}x{board_size} board with model: {time_taken} seconds")


if __name__ == '__main__':
    unittest.main()