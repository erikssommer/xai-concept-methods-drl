import unittest

import env

class TestRandomPlay(unittest.TestCase):
    # Create the go environment
    go_env = env.GoEnv(size=9, komi=6.5, reward_method='real')

    # Reset the environment
    go_env.reset()

    # Play a random game
    terminated = False

    while not terminated:
        action = go_env.uniform_random_action()
        state, reward, terminated, info = go_env.step(action)

    
    # Render the final board
    go_env.render()


if __name__ == '__main__':
    unittest.main()