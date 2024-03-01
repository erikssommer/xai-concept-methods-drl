from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf

def perform_regression(points, targets, validation_points, validation_targets, is_binary, epochs=50, dynamic=False, verbose=0):
    if is_binary:
        return perform_logistic_regression(
            points, 
            targets, 
            validation_points, 
            validation_targets,
            epochs=epochs,
            dynamic=dynamic,
            verbose=verbose
        )
    else:
        return perform_linear_regression(
            points,
            targets, 
            validation_points, 
            validation_targets,
            dynamic=dynamic,
        )

def perform_logistic_regression(points, targets, validation_points, validation_targets, epochs, dynamic, verbose):
    """
    Classification using logistic regression
    """
    inputs = tf.keras.layers.Input((points.shape[1]))
    output = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))(inputs)

    model = tf.keras.Model(inputs, output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    with tf.device('/GPU:0'):
        if dynamic:
            model.fit(points, targets, epochs=epochs, verbose=0)
        else:
            model.fit(points, targets, validation_data=(validation_points, validation_targets), epochs=epochs, verbose=verbose)

    train_preds = model.predict(points) > 0.5
    print(binary_accuracy_metric(targets, train_preds))
    if dynamic:
        return binary_accuracy_metric(targets, train_preds)
    val_preds = model.predict(validation_points) > 0.5
    return binary_accuracy_metric(validation_targets, val_preds)


def perform_linear_regression(points, targets, validation_points, validation_targets, dynamic):
    """
    Regression using linear regression
    """
    model = linear_model.LinearRegression()
    model = model.fit(points, targets)
    if dynamic:
        predictions = model.predict(points)
        return r2_score(targets, predictions)
    
    predictions = model.predict(validation_points)
    return r2_score(validation_targets, predictions)


def binary_accuracy_metric(targets, predictions):
    return 2 * (((targets == np.squeeze(predictions)).sum() / len(targets)) - 0.5)