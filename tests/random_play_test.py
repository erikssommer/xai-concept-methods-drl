import gym
import unittest

# Set the logging level
gym.logger.set_level(40)

class TestRandomPlay(unittest.TestCase):
    # Create the go environment
    go_env = gym.make('gym_go:go-v0', size=9, komi=6.5, reward_method='real')

    # Reset the environment
    go_env.reset()

    # Play a random game
    terminated = False

    while not terminated:
        action = go_env.uniform_random_action()
        state, reward, terminated, info = go_env.step(action)

        # Render the board
        go_env.render()

    # Close the environment
    go_env.close()


if __name__ == '__main__':
    unittest.main()