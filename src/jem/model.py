"""
Class for the joint embedding model
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm


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
        self.learning_rate = learning_rate

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

            # Concatenate the two embeddings and create the model
            self.model = tf.keras.Model(inputs=[board_inputs, explanation_inputs], 
                                        outputs=[state_final, explanation_final])

            if summary:
                self.model.summary()

    def fit(self, train_states, train_explinations, train_labels, val_states, val_explinations, val_labels, batch_size=32, epochs=10):
        # Define a custom training loop using the loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        
        @tf.function
        def loss_fn(y, state_embed, concept_embed):
            # Split the concatinated pred tensor into state_embed and concept_embed
            batch_size = tf.shape(state_embed)[0]
            difference = state_embed - concept_embed 
            l2_norm = tf.norm(difference, axis=1, ord=2)
            loss = tf.reduce_sum((l2_norm - y) ** 2) / tf.cast(batch_size, dtype=tf.float32)
            return loss

        @tf.function
        def train_step(states, explinations, labels):
            with tf.GradientTape() as tape:
                state_embed, concept_embed = self.model([states, explinations], training=True)
                loss = loss_fn(labels, state_embed, concept_embed)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            train_loss(loss)

        @tf.function
        def val_step(states, explinations, labels):
            state_embed, concept_embed = self.model([states, explinations], training=False)
            loss = loss_fn(labels, state_embed, concept_embed)
            val_loss(loss)

        # Train the model
        bar = tqdm(range(epochs))
        for _ in bar:
            train_loss.reset_states()
            val_loss.reset_states()

            for i in range(0, len(train_states), batch_size):
                train_step(train_states[i:i+batch_size], train_explinations[i:i+batch_size], train_labels[i:i+batch_size])

            for i in range(0, len(val_states), batch_size):
                val_step(val_states[i:i+batch_size], val_explinations[i:i+batch_size], val_labels[i:i+batch_size])

            bar.set_description(f'Loss: {train_loss.result()}, Val Loss: {val_loss.result()}')

    def predict(self, state: np.ndarray, explination: np.ndarray):
        if len(state.shape) == 3:
            state = state.reshape((1, *state.shape))

        if len(explination.shape) == 1:
            explination = explination.reshape((1, *explination.shape))

        with tf.device('/device:CPU:0'):
            return self.model([state, explination], training=False)

    def save_model(self, path):
        self.model.save(path)
