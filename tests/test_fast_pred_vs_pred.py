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
from mcts import MCTS
import env

class TestFastPredVsPred(unittest.TestCase):
    def test_equal(self):
        # Loading model
        board_size = 7
        komi = 9.5
        model_path = f"../models/saved_sessions/resnet/board_size_{board_size}/falcon/net_500.keras"

        model = ResNet(board_size, model_path)
        fast_model = FastPredictor(LiteModel.from_keras_model(model.model))

        # Creating input
        go_env = env.GoEnv(board_size, komi)
        go_env.reset()
        state = go_env.canonical_state()

        # Using mcts to generate a state
        mcts = MCTS(state, 1000, board_size, board_size * board_size * 2, fast_model, komi=komi)

        mcts.search()

        # Move to a leaf node
        root = mcts.root

        # Loop through the children and select the one with the highest visit count
        while root.is_expanded() and root.predict_state_rep is not None:
            highest_visits = -1
            best_child = None
            for child in root.children:
                if child.n_visit_count > highest_visits:
                    highest_visits = child.n_visit_count
                    best_child = child
            root = best_child
        
        state = root.parent.predict_state_rep


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
        