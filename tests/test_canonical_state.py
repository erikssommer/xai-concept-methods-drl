import unittest

import os
# Add the src folder to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

import env

class TestCanonicalState(unittest.TestCase):
    def test_canonical_state(self):
        go_env = env.GoEnv(4,0.5)
        go_env.reset()

        game_over = False

        while not game_over:
            print(go_env.canonical_state())
            print("--------------------")
            action = go_env.uniform_random_action()
            _, _, game_over, _ = go_env.step(action)
        
        print("Winner: ", go_env.winner())

if __name__ == '__main__':
    unittest.main()
