"""
Class for the joint embedding model
"""

import tensorflow as tf
import numpy as np


class JointEmbeddingModel:
    def __init__(self,
                 vocab_size,
                 max_sent_len,
                 board_size,
                 learning_rate=0.001,
                 input_state_embed=64,
                 hidden_state_embed=32,
                 output_state_embed=16,
                 exp_embed=32,
                 output_exp_embed=16,
                 load_path: str = None,
                 summary: bool = True):
        
        num_channels = 5

        if load_path:
            self.model = tf.keras.models.load_model(load_path)
        else:
            # Convolutional layers for the board
            board_inputs = tf.keras.Input(shape=(num_channels, board_size, board_size))
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(board_inputs)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(units=input_state_embed, activation='relu')(x)
            x = tf.keras.layers.Dense(units=hidden_state_embed, activation='relu')(x)
            state_final = tf.keras.layers.Dense(units=output_state_embed, name='final_state')(x)

            # Embedding and LSTM layers for the textual explanation
            explanation_inputs = tf.keras.Input(shape=(max_sent_len,))
            x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=exp_embed, mask_zero=True)(explanation_inputs)
            x = tf.keras.layers.LSTM(units=output_exp_embed)(x)
            explanation_final = tf.keras.layers.ReLU(name='final_explanation')(x)

            combined_outputs = tf.keras.layers.concatenate([state_final, explanation_final], name="combined_output")
            # Concatenate the two embeddings and create the model
            self.model = tf.keras.Model(inputs=[board_inputs, explanation_inputs], 
                                        outputs=[state_final, explanation_final, combined_outputs])

            if summary:
                self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'combined_output': self.loss_fn}
        )

    def loss_fn(self, y, pred):
        # Split the concatinated pred tensor into state_embed and concept_embed
        state_embed, concept_embed = tf.split(pred, 2, axis=1)

        batch_size = tf.shape(state_embed)[0]
        difference = state_embed - concept_embed 
        l2_norm = tf.norm(difference, axis=1, ord=2)

        loss = tf.reduce_sum((l2_norm - y) ** 2) / tf.cast(batch_size, dtype=tf.float32)

        return loss

    def fit(self, state, explination, y, batch_size, epochs):
        with tf.device('/device:GPU:0'):
            self.model.fit(x=[state, explination], y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, state: np.ndarray, explination: np.ndarray):
        if len(state.shape) == 3:
            state = state.reshape((1, *state.shape))

        if len(explination.shape) == 1:
            explination = explination.reshape((1, *explination.shape))

        with tf.device('/device:CPU:0'):
            return self.model([state, explination], training=False)

    def save_model(self, path):
        self.model.save(path)
