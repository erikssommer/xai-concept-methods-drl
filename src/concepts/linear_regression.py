from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf

def perform_regression(points, targets, validation_points, validation_targets, is_binary):
    if is_binary:
        return perform_logistic_regression(points, targets, validation_points, validation_targets)
    else:
        return perform_linear_regression(points, targets, validation_points, validation_targets)


def perform_logistic_regression(points, targets, validation_points, validation_targets):
    inputs = tf.keras.layers.Input((points.shape[1]))
    output = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))(inputs)

    model = tf.keras.Model(inputs, output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    model.fit(points, targets, validation_data=(validation_points, validation_targets), epochs=50)

    train_preds = model.predict(points) > 0.5
    val_preds = model.predict(validation_points) > 0.5
    print(binary_accuracy_metric(targets, train_preds))
    return binary_accuracy_metric(validation_targets, val_preds)


def perform_linear_regression(points, targets, validation_points, validation_targets):
    model = linear_model.LinearRegression()
    model = model.fit(points, targets)
    predictions = model.predict(validation_points)
    return r2_score(validation_targets, predictions)


def binary_accuracy_metric(targets, predictions):
    return 2 * (((targets == np.squeeze(predictions)).sum() / len(targets)) - 0.5)