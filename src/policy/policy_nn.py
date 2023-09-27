import tensorflow as tf
import numpy as np
from utils import config

class ActorCriticNet(tf.keras.Model):
    def __init__(self, input_shape, output, init=True):
        super(ActorCriticNet, self).__init__()

        BLOCK_FILTER_SIZE = 32

        if not init:
            return
        
        # Input
        self.position_input = tf.keras.Input((input_shape))
        
        # Residual block
        base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same", name="res_block_output_base")(self.position_input)
        base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(base)
        base = tf.keras.layers.Conv2D(BLOCK_FILTER_SIZE, (3, 3), activation="elu", padding="same")(base)

        # Policy head
        policy = tf.keras.layers.Conv2D(output, (1, 1), activation="elu", padding="same")(base)
        policy = tf.keras.layers.Flatten()(policy)
        policy = tf.keras.layers.Dense(output, activation="relu")(policy)
        policy_output = tf.keras.layers.Softmax(name="policy_output")(policy)

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
        
        return self.model.fit(states, [distributions, values], verbose=0, epochs=epochs, batch_size=128)
        
    def predict(self, boards):
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))
        
        res = self.model(boards, training=False)
        
        policies, values = res
        
        return policies, values
    
    def predict_multi(self, boards):
        res = self.model.predict(np.array(boards))

        policies, values = res

        return policies, values
    
