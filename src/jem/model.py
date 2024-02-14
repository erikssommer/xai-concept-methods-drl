"""
Class for the joint embedding model
"""

import tensorflow as tf

from utils import config

class JointEmbeddingModel:
    def __init__(self, vocab_size, numChannels=1, input_state_embed=64, hidden_state_embed=32, output_state_embed=16, exp_embed=32, output_exp_embed=16, load_path: str = None, summary: bool = True):
        board_size = config.board_size
        if load_path:
            self.model = tf.keras.models.load_model(load_path)
        else:
            board_inputs = tf.keras.Input(shape=(board_size, board_size, numChannels))
            x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(board_inputs)
            x = tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(units=input_state_embed, activation='relu')(x)
            x = tf.keras.layers.Dense(units=hidden_state_embed, activation='relu')(x)
            state_final = tf.keras.layers.Dense(units=output_state_embed)(x)

            explanation_inputs = tf.keras.Input(shape=(11,))
            x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=exp_embed, mask_zero=True)(explanation_inputs)
            x = tf.keras.layers.LSTM(units=output_exp_embed)(x)
            explanation_final = tf.keras.layers.ReLU()(x)

            self.model = tf.keras.Model(inputs=[board_inputs, explanation_inputs], outputs=[state_final, explanation_final])

        if summary:
            self.model.summary()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss=self.loss_fn,
            metrics=['accuracy'])


    def loss_fn(self, state_embed, concept_embed, y):
        batch_size = tf.shape(state_embed)[0]
        difference = state_embed - concept_embed 
        l2_norm = tf.norm(difference, axis=1, ord=2)
        #l2_norm = tf.sqrt(tf.reduce_sum(((state_embed * concept_embed)) ** 2, axis=1))
        loss = tf.reduce_sum((l2_norm - y) ** 2) / tf.cast(batch_size, dtype=tf.float32)

        return loss
    
    def fit(self, state, explination, y, batch_size, epochs, validation_data):
        with tf.device('/device:GPU:0'):
            self.model.fit([state, explination], y, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def predict(self, state, explination):
        return self.model.predict([state, explination])
    
    def save_model(self, path):
        self.model.save(path)