import gym
import tensorflow as tf
import numpy as np

from models.policy_nn import ActorCriticNet
from utils.read_config import config
from game.data import GoVars, GoGame

# Main method
if __name__ == "__main__":
    # Set the logging level
    gym.logger.set_level(40)

    # Create the go environment
    go_env = gym.make('gym_go:go-v0', size=config.board_size, komi=0, reward_method='real')

    # Reset the environment
    go_env.reset()

    model: ActorCriticNet = tf.keras.models.load_model('../models/actor_critic_net_10')

    # Play a random game
    terminated = False

    invalid_predictions = 0
    num_predictions = 0

    while not terminated:
        if go_env.turn() == 0:
            # Get the action from the neural network
            distribution, value = model.predict(np.array([go_env.canonical_state()]))
            action = np.argmax(distribution[0])
            # Test if move is valid
            valid_actions = go_env.valid_moves()

            if valid_actions[action] != 1:
                invalid_predictions += 1
                action = go_env.uniform_random_action()
            
            num_predictions += 1
            state, reward, terminated, info = go_env.step(action)
        else:
            action = go_env.uniform_random_action()
            state, reward, terminated, info = go_env.step(action)

        # Render the board
        go_env.render()

    print("Number of predictions: {}".format(num_predictions))
    print("Invalid predictions: {}".format(invalid_predictions))

    # Close the environment
    go_env.close()