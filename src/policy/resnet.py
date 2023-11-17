import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import config
from env import gogame, govars
import utils
from .basenet import BaseNet


class ResNet(BaseNet):
    def __init__(self, board_size, load_path=None, summary=True):
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

        if load_path:
            self.model = tf.keras.models.load_model(load_path)
        else:
            position_input = tf.keras.Input((4, board_size, board_size))
            base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same", name="res_block_output_base")(position_input)
            block_amount = config.resnet_blocks
            for block in range(block_amount):
                # Convolve "input" twice
                conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(base)
                conv = tf.keras.layers.BatchNormalization()(conv)
                conv = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(conv)
                conv = tf.keras.layers.BatchNormalization()(conv)
                # Add and relu residues to the doubly convolved layer
                conv_with_residues = tf.keras.layers.Add()([base, conv])
                base = tf.keras.layers.ELU(name="res_block_output_{}".format(block))(conv_with_residues)

            # Policy head
            policy = tf.keras.layers.Conv2D(self.output, (1, 1), activation="elu", padding="same")(base)
            policy = tf.keras.layers.Flatten()(policy)
            policy_output = tf.keras.layers.Dense(self.output, activation="softmax", name="policy_output")(policy)

            # Value head
            val = tf.keras.layers.Conv2D(16, (1, 1), name="value_conv", activation="elu", padding="same")(base)
            val = tf.keras.layers.Flatten()(val)
            # val = keras.layers.Dense(256, activation="elu")(val)
            value_output = tf.keras.layers.Dense(1, name="value_output", activation="tanh")(val)

            self.model = tf.keras.Model(position_input, [policy_output, value_output])
            
            if summary:
                self.model.summary()
        
        self.model.compile(
            loss={"policy_output": tf.keras.losses.CategoricalCrossentropy(), "value_output": tf.keras.losses.MeanSquaredError()},
            loss_weights={"policy_output": 1.0, "value_output": 1.0},
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            metrics=["accuracy"]
            )

    def get_all_activation_values(self, boards, keyword="conv2d"):
        """Returns a model that gives the activations from resnet-blocks"""
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))

        # All inputs
        inp = self.model.input
        # All outputs of the residual blocks
        outputs = [layer.output for layer in self.model.layers if keyword in layer.name]
        functor = tf.keras.backend.function([inp], outputs)

        BATCH_SIZE = 32
        all_layer_outs = []
        for i in tqdm(range(0, boards.shape[0], BATCH_SIZE)):
            layer_outs = functor([boards[i:i + BATCH_SIZE]])
            all_layer_outs.append(layer_outs)

        return all_layer_outs

    def fit(self, states, distributions, values, callbacks=None, epochs=10):
        if config.use_gpu:
            with tf.device('/gpu:0'):
                return self.model.fit(states, [distributions, values], verbose=0, epochs=epochs, batch_size=128, callbacks=callbacks)
        else:
            return self.model.fit(states, [distributions, values], verbose=0, epochs=epochs, batch_size=128, callbacks=callbacks)
    
    # Define a prediction function
    def predict(self, state, value_only=False):
        state_copy = state.copy()

        # Current players stones is allways first layer
        if gogame.turn(state) == govars.WHITE:
            channels = np.arange(govars.NUM_CHNLS)
            channels[govars.BLACK] = govars.WHITE
            channels[govars.WHITE] = govars.BLACK
            state = state[channels]

        # Remove array index 3 and 5 from the current state making it an shape of (4, 5, 5)
        state = np.delete(state, [3, 5], axis=0)

        if len(state.shape) == 3:
            state = np.reshape(state, (1, *state.shape))
        if config.use_gpu:
            with tf.device('/gpu:0'):
                res = self.model(state, training=False)
        else:
            res = self.model(state, training=False)
        
        policy, value = res

        # Get the policy array and value number from the result
        policy = policy[0]
        value = value[0][0]

        if value_only:
            return value

        policy = self.mask_invalid_moves(policy, state_copy)
        
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
        
        return policy
    
    def best_action(self, state, greedy_move=False, alpha=config.alpha):
        policy, _ = self.predict(state)

        if greedy_move:
            return np.argmax(policy)

        if alpha > np.random.random():
            # Selecting move randomly, but weighted by the distribution (0 = argmax, 1 = probablistic)
            return np.random.choice(len(policy), p=policy)

        # Selecting move greedily
        return np.argmax(policy)
    
    def value_estimation(self, state):
        value = self.predict(state, value_only=True)
        return value
    
    def save_model(self, path):
        self.model.save(path)