import tensorflow as tf
import numpy as np
from utils import config
from env import gogame
import utils
from tqdm import tqdm
from .basenet import BaseNet

class ConvNet(BaseNet):
    def __init__(self, board_size, load_path=None, summary=True):
        """
        Convolutional neural network for the actor-critic policy.

        Args:
            board_size (int): The size of the board.
            load_path (str, optional): The path to the model to load. Defaults to None.
            summary (bool, optional): Whether to print the summary of the model. Defaults to True.
        """

        BLOCK_FILTER_SIZE = config.convnet_filters

        self.board_size = board_size
        self.load_path = load_path
        self.output = board_size ** 2 + 1

        if load_path:
            self.model = tf.keras.models.load_model(load_path)
        else:
            # Input
            self.position_input = tf.keras.Input(shape=(2, self.board_size, self.board_size))
            
            # Residual block
            base = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="res_block_output_base")(self.position_input)
            base = tf.keras.layers.Conv2D(48, (3, 3), activation="relu", padding="same")(base)
            base = tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same")(base)
            base = tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same")(base)

            # Flatten the base
            base = tf.keras.layers.Flatten()(base)

            base = tf.keras.layers.Dense(128, activation="relu")(base)
            base = tf.keras.layers.Dense(64, activation="relu")(base)

            # Policy head
            policy_output = tf.keras.layers.Dense(self.output, activation="softmax", name="policy_output")(base)

            # Value head
            value_output = tf.keras.layers.Dense(1, activation="tanh", name="value_output")(base)

            self.model = tf.keras.Model(self.position_input, [policy_output, value_output])
            
            if summary:
                self.model.summary()

        self.model.compile(
            loss={"policy_output": tf.keras.losses.CategoricalCrossentropy(), "value_output": tf.keras.losses.MeanSquaredError()},
            loss_weights={"policy_output": 1.0, "value_output": 1.0},
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            metrics=["accuracy"]
            )
    
    def get_all_activation_values(self, boards, keyword="conv"):
        """Returns a list of all the activation values for each layer in the model"""
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))

        # All inputs
        inp = self.model.input
        # All outputs of the conv blocks
        outputs = [layer.output for layer in self.model.layers if keyword in layer.name]
        functor = tf.keras.backend.function([inp], outputs)

        BATCH_SIZE = 32
        all_layer_outs = []
        for i in tqdm(range(0, boards.shape[0], BATCH_SIZE), desc="Getting activation outputs"):
            layer_outs = functor([boards[i:i + BATCH_SIZE]])
            all_layer_outs.append(layer_outs)

        return all_layer_outs

    def fit(self, states, distributions, values, callbacks=None, epochs=10):
        return self.model.fit(states, [distributions, values], verbose=0, epochs=epochs, batch_size=128, callbacks=callbacks)
    
    # Define a prediction function
    def predict(self, state, player, value_only=False, mock_data=False):

        if mock_data:
            policy = np.random.random(self.board_size ** 2 + 1)
            policy = self.mask_invalid_moves(policy, state)

            value = np.random.random()
            return policy, value

        state_copy = state.copy()

        # Remove array index 3 and 5 from the current state making it an shape of (4, 5, 5)
        state = np.delete(state, [2,3,4,5], axis=0)
        #if player == 1:
            #state[2] = np.ones((self.board_size, self.board_size))

        if len(state.shape) == 3:
            state = np.reshape(state, (1, *state.shape))

        res = self.model(state, training=False)

        policy, value = res

        # Get the policy array and value number from the result
        policy = policy[0]
        value = value[0][0]

        if value_only:
            return value

        policy = self.mask_invalid_moves(policy, state_copy)

        del state
        del state_copy
        
        return policy, value
    
    def mask_invalid_moves(self, policy, state):
        # Get invalid moves
        valid_moves = gogame.valid_moves(state)

        # Mask the invalid moves
        policy = policy * valid_moves

        # Reduce to 8 decimal places
        policy = np.round(policy, 8)
    
        # Normalize the policy
        policy = utils.normalize(policy)

        del valid_moves
        
        return policy
    
    def best_action(self, state, player, greedy_move=False, alpha=config.alpha):
        policy, value = self.predict(state, player)

        print("Value: ", value)

        if greedy_move:
            return np.argmax(policy)

        # Selecting move randomly, but weighted by the distribution (0 = argmax, 1 = probablistic)
        return np.random.choice(len(policy), p=policy)
    
    def value_estimation(self, state):
        return self.predict(state, value_only=True)
    
    def save_model(self, path):
        self.model.save(path)
