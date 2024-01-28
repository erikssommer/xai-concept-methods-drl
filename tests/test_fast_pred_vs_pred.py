import unittest
import numpy as np
import time

import os
# Add the src folder to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from policy import ConvNet, ResNet
from policy import FastPredictor
from policy import LiteModel
import env

class TestFastPredVsPred(unittest.TestCase):
    def test_equal(self):
        # Loading model
        board_size = 5
        komi = 0.5
        model_path = f"../models/training/board_size_{board_size}/net_0.keras"

        model = ConvNet(board_size, model_path)
        fast_model = FastPredictor(LiteModel.from_keras_model(model.model))

        # Creating input
        go_env = env.GoEnv(board_size, komi)
        go_env.reset()

        current_player = 0
        # Play some random moves
        for _ in range(4):
            action = go_env.uniform_random_action()
            _, _, game_over, _ = go_env.step(action)
            current_player = 1 - current_player

        # Getting the state
        state = go_env.canonical_state()

        state = np.delete(state, [2,3,4,5], axis=0)

        #if current_player == 1:
            #print("Current player is 1")
            #state[2] = np.ones((board_size, board_size))

        state_slow = np.reshape(state, (1, *state.shape))

        # Predicting
        # Set a timer for the slow model
        start = time.time()
        #dist, value = model.model.predict(state_slow) even slower!!! Don't use this
        dist, value = model.model(state_slow)
        end = time.time()
        print("Slow model time: ", end - start)

        # Set a timer for the fast model
        start = time.time()
        fast_dist, fast_value = fast_model.model.predict_single(state)
        end = time.time()
        print("Fast model time: ", end - start)

        print("Dist: ", dist)
        print("Fast dist: ", fast_dist)
        print("Value: ", value)
        print("Fast value: ", fast_value)

if __name__ == '__main__':
    unittest.main()
        