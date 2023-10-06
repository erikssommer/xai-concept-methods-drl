import tensorflow as tf
import os
import time
from utils import config

def tensorboard_setup():
    # Delete the ../tensorboard_logs directory if it exists
    if os.path.exists('../tensorboard_logs'):
        os.system('rm -rf ../tensorboard_logs')

    # Create a log directory with a timestamp
    logdir = f'../{config.log_dir}/' + time.strftime("%Y%m%d-%H%M%S")

    # Create a TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    return tensorboard_callback, logdir

def write_to_tensorboard(history, episode, logdir):
     # Add the metrics to TensorBoard
    with tf.summary.create_file_writer(logdir).as_default():
        for loss in ["loss", "value_output_loss", "policy_output_loss"]:
            tf.summary.scalar(name=loss, data=history.history[loss][0], step=episode)
        for acc in ["value_output_accuracy", "policy_output_accuracy"]:
            tf.summary.scalar(name=acc, data=history.history[acc][0], step=episode)