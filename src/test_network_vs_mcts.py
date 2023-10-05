import unittest
import numpy as np
import env
from policy import ActorCriticNet
import os
from utils import config
from mcts import MCTS

class TestMCTSvsNetwork(unittest.TestCase):
    def test_network_as_black_vs_mcts(self):
        go_env = env.GoEnv(size=5)

        go_env.reset()

        # Find the model with the highest number in the name from the models/board_size_5 folder
        path = f'../models/board_size_{config.board_size}/'

        folders = os.listdir(path)

        # Sort the folders by the number in the name
        sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-1].strip('.keras')))

        # Get the last folder
        path = path + sorted_folders[-1]

        print("Loading model from: {}".format(path))

        actor_net = ActorCriticNet(5, path)
        actor_mcts = MCTS(go_env.canonical_state(), 1, 1, 100, 5, 5**2*5)

        games = 10
        winns = 0

        for _ in range(games):
            go_env.reset()

            game_over = False

            while not game_over:
                if go_env.turn() == 0:
                    distribution, _ = actor_net.predict(go_env.state())
                    action = np.argmax(distribution[0])
                    _, _, game_over, _ = go_env.step(action)
                else:
                    actor_mcts.set_root(go_env.state())
                    best_action_node, _, distribution = actor_mcts.search()
                    _, _, game_over, _ = go_env.step(best_action_node.action)
            
            winner = go_env.winning()

            if winner == 1:
                winns += 1

        
        win_probability = winns / games

        print("Win probability as black: {}".format(win_probability))
        assert win_probability >= 0.9
                

    def test_network_as_white_vs_mcts(self):
        go_env = env.GoEnv(size=5)

        go_env.reset()

       # Find the model with the highest number in the name from the models/board_size_5 folder
        path = f'../models/board_size_{config.board_size}/'

        folders = os.listdir(path)

        # Sort the folders by the number in the name
        sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-1].strip('.keras')))

        # Get the last folder
        path = path + sorted_folders[-1]

        print("Loading model from: {}".format(path))

        actor_net = ActorCriticNet(5, path)
        actor_mcts = MCTS(go_env.canonical_state(), 1, 1, 100, 5, 5**2*5)

        games = 10
        winns = 0

        for _ in range(games):
            go_env.reset()

            game_over = False

            while not game_over:
                if go_env.turn() == 1:
                    distribution, _ = actor_net.predict(go_env.state())
                    action = np.argmax(distribution[0])
                    _, _, game_over, _ = go_env.step(action)
                else:
                    actor_mcts.set_root(go_env.state())
                    best_action_node, _, distribution = actor_mcts.search()
                    _, _, game_over, _ = go_env.step(best_action_node.action)
            
            winner = go_env.winning()

            if winner == -1:
                winns += 1

        
        win_probability = winns / games

        print("Win probability as white: {}".format(win_probability))
        assert win_probability >= 0.9


if __name__ == '__main__':
    unittest.main()