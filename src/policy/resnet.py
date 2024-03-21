import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import config
import utils
from .basenet import BaseNet
from typing import Tuple


class ResNet(BaseNet):
    def __init__(self, board_size: int, load_path: str = None, summary: bool = True):
        """
        Residual neural network for the actor-critic policy.

        Args:
            board_size (int): The size of the board.
            load_path (str, optional): The path to the model to load. Defaults to None.
            summary (bool, optional): Whether to print the summary of the model. Defaults to True.
        """

        BLOCK_FILTER_SIZE = config.resnet_filters

        self.board_size = board_size
        self.load_path = load_path
        self.output = board_size ** 2 + 1
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size

        if load_path:
            self.model = tf.keras.models.load_model(load_path)
        else:
            self.position_input = tf.keras.Input(
                shape=(5, board_size, board_size))

            # Residual block
            base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="relu", padding="same", name="res_block_output_base")(self.position_input)

            # First
            conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), padding="same")(base)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.ReLU()(conv)
            conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), padding="same")(conv)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.ReLU()(conv)

            # Add skip connection
            conv_res = tf.keras.layers.Add()([base, conv])
            base = tf.keras.layers.ReLU(name="res_block_1_relu")(conv_res)

            conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), padding="same")(base)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.ReLU()(conv)
            conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), padding="same")(conv)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.ReLU()(conv)

            # Add skip connection
            conv_res = tf.keras.layers.Add()([base, conv])
            base = tf.keras.layers.ReLU(name="res_block_2_relu")(conv_res)

            conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), padding="same")(base)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.ReLU()(conv)
            conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), padding="same")(conv)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.ReLU()(conv)

            # Add skip connection
            conv_res = tf.keras.layers.Add()([base, conv])
            base = tf.keras.layers.ReLU(name="res_block_3_relu")(conv_res)

            # Policy head
            policy = tf.keras.layers.Conv2D(2, (1, 1), padding="same")(base)
            policy = tf.keras.layers.BatchNormalization()(policy)
            policy = tf.keras.layers.ReLU()(policy)
            policy = tf.keras.layers.Flatten()(policy)
            policy_output = tf.keras.layers.Dense(self.output, activation="softmax", name="policy_output")(policy)

            # Value head
            value = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(base)
            value = tf.keras.layers.BatchNormalization()(value)
            value = tf.keras.layers.ReLU()(value)
            value = tf.keras.layers.Flatten()(value)
            value = tf.keras.layers.Dense(BLOCK_FILTER_SIZE, activation="relu")(value)
            value_output = tf.keras.layers.Dense(1, activation="tanh", name="value_output")(value)

            self.model = tf.keras.Model(
                self.position_input, [policy_output, value_output])

            if summary:
                self.model.summary()

        self.model.compile(
            loss={"policy_output": tf.keras.losses.CategoricalCrossentropy(),
                  "value_output": tf.keras.losses.MeanSquaredError()},
            loss_weights={"policy_output": 1.0, "value_output": 1.0},
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate),
            metrics=["accuracy"]
        )

    def get_all_activation_values(self, boards, keyword: str = "conv") -> list:
        """Returns a list of all the activation values for each layer in the model"""
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))

        # All inputs
        inp = self.model.input
        # All outputs of the conv blocks
        outputs = [
            layer.output for layer in self.model.layers if keyword in layer.name]
        functor = tf.keras.backend.function([inp], outputs)

        BATCH_SIZE = 32
        all_layer_outs = []
        for i in tqdm(range(0, boards.shape[0], BATCH_SIZE), desc="Getting activation outputs"):
            layer_outs = functor([boards[i:i + BATCH_SIZE]])
            all_layer_outs.append(layer_outs)

        return all_layer_outs

    def fit(self, states, distributions, values, callbacks=None, epochs=10) -> tf.keras.callbacks.History:
        with tf.device("/GPU:0"):
            return self.model.fit(states, [distributions, values], verbose=0, shuffle=True, epochs=epochs, batch_size=self.batch_size, callbacks=callbacks)

    # Define a prediction function
    def predict(self, state: np.ndarray, valid_moves: np.ndarray, value_only: bool = False) -> Tuple[np.ndarray, float]:
        """Predict the policy and value of a state"""
        if len(state.shape) == 3:
            state = np.reshape(state, (1, *state.shape))

        with tf.device("/CPU:0"):
            res = self.model(state, training=False)

        policy, value = res

        # Get the policy array and value number from the result
        policy = policy[0]
        value = value[0][0]

        if value_only:
            return value

        policy = self.mask_invalid_moves(policy, valid_moves)

        del state

        return policy, value

    def mask_invalid_moves(self, policy: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:

        # Mask the invalid moves
        policy = policy * valid_moves

        # Convert to 8 decimals
        policy = np.round(policy, 8)

        # Normalize the policy
        policy = utils.normalize(policy)

        return policy

    def best_action(self, state: np.ndarray, valid_moves: np.ndarray, greedy_move: bool = False, alpha: float = None) -> Tuple[int, float]:
        policy, value = self.predict(state, valid_moves)

        value = value.numpy()

        if greedy_move:
            return np.argmax(policy), value

        if alpha and np.random.random() < alpha:
            # Selecting move randomly, but weighted by the distribution (0 = argmax, 1 = probablistic)
            return np.argmax(policy), value

        # Selecting move randomly, but weighted by the distribution (0 = argmax, 1 = probablistic)
        return np.random.choice(len(policy), p=policy), value

    def value_estimation(self, state, valid_moves):
        return self.predict(state, valid_moves, value_only=True)

    def save_model(self, path):
        self.model.save(path)
