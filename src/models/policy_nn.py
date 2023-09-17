import tensorflow as tf
import numpy as np
from game.data import GoGame

class ActorCriticNet(tf.keras.Model):
    def __init__(self, size):
        action_size = GoGame.action_size(board_size=size)
        super(ActorCriticNet, self).__init__()

        self.action_size = action_size

        self.act_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(action_size)
        ])

        self.crit_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.game_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, kernel_size=1),  # Assuming `self.channels` is 6
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(6 * (size ** 2 + 1), kernel_size=1)
        ])
    
    