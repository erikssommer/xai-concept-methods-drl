import unittest
import numpy as np
import env
from policy import ActorCriticNet
import os
from utils import config

class TestMCTSvsRandom(unittest.TestCase):
    def test_network_as_black_vs_random(self):
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

        games = 100
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
                    action = go_env.uniform_random_action()
                    _, _, game_over, _ = go_env.step(action)
            
            winner = go_env.winning()

            if winner == 1:
                winns += 1

        
        win_probability = winns / games

        print("Win probability as black: {}".format(win_probability))
        assert win_probability >= 0.9
                

    def test_network_as_white_vs_random(self):
        go_env = env.GoEnv(size=5)

        go_env.reset()

        path = '../models/board_size_5/net_10.keras'
        actor_net = ActorCriticNet(5, path)

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
                    action = go_env.uniform_random_action()
                    _, _, game_over, _ = go_env.step(action)
            
            winner = go_env.winning()

            if winner == -1:
                winns += 1

        
        win_probability = winns / games

        print("Win probability as white: {}".format(win_probability))
        assert win_probability >= 0.9


if __name__ == '__main__':
    unittest.main()