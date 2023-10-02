import tensorflow as tf
import numpy as np
from utils import config
from env import gogame
import utils

class ActorCriticNet:
    def __init__(self, board_size, load_path=None):

        BLOCK_FILTER_SIZE = 32

        self.board_size = board_size
        self.load_path = load_path
        self.output = board_size ** 2 + 1

        if load_path:
            self.model = tf.keras.models.load_model(load_path)
        else:
            # Input
            self.position_input = tf.keras.Input(shape=(6, self.board_size, self.board_size))
            
            # Residual block
            base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same", name="res_block_output_base")(self.position_input)
            base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(base)
            base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(base)

            # Policy head
            policy = tf.keras.layers.Conv2D(self.output, (1, 1), activation="elu", padding="same")(base)
            policy = tf.keras.layers.Flatten()(policy)
            policy_output = tf.keras.layers.Dense(self.output, activation="softmax", name="policy_output")(policy)

            # Value head
            val = tf.keras.layers.Conv2D(16, (1, 1), name="value_conv", activation="elu", padding="same")(base)
            val = tf.keras.layers.Flatten()(val)
            value_output = tf.keras.layers.Dense(1, name="value_output", activation="tanh")(val)

            self.model = tf.keras.Model(self.position_input, [policy_output, value_output])

        self.model.summary()
        self.model.compile(
            loss={"policy_output": tf.keras.losses.CategoricalCrossentropy(), "value_output": tf.keras.losses.MeanSquaredError()},
            loss_weights={"policy_output": 1.0, "value_output": 1.0},
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate))

    def fit(self, states, distributions, values, epochs=10):
        if config.use_gpu:
            with tf.device('/gpu:0'):
                return self.model.fit(states, [distributions, values], verbose=0, epochs=epochs, batch_size=128)
        else:
            return self.model.fit(states, [distributions, values], verbose=0, epochs=epochs, batch_size=128)
    
    # Define a prediction function
    def predict(self, state):
        state_copy = state.copy()
        if len(state.shape) == 3:
            state = np.reshape(state, (1, *state.shape))
        if config.use_gpu:
            with tf.device('/gpu:0'):
                res = self.model(state, training=False)
        else:
            res = self.model(state, training=False)
        
        policy, values = res

        policy = self.mask_invalid_moves(policy, state_copy)
        
        return policy, values
    
    def mask_invalid_moves(self, policy, state):
        # Get invalid moves
        valid_moves = gogame.valid_moves(state)

        # Mask the invalid moves
        policy = policy * valid_moves
    
        # Normalize the policy
        policy = utils.normalize(policy)
        
        return policy
    
    def save_model(self, path):
        self.model.save(path)
