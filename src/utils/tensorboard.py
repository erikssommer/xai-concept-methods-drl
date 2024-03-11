import tensorflow as tf
import os
import datetime

def tensorboard_setup(model_type, reward_function_type, clear_tensorboard):
    reward_function_type = f'{reward_function_type}_reward_function'
    # Delete the ../tensorboard_logs directory if it exists
    if os.path.exists('../tensorboard_logs'):
        # Test if the direcory contains files containing the reward function type
        for folder in os.listdir('../tensorboard_logs'):
            if clear_tensorboard:
                # Remove the file
                os.system(f'rm -rf ../tensorboard_logs/{folder}')
            elif reward_function_type and model_type in folder:
                # Remove the file
                os.system(f'rm -rf ../tensorboard_logs/{folder}')
    
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create a log directory with a timestamp
    logdir = f'../tensorboard_logs/{model_type}-{reward_function_type}-{date}'

    # Create a TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    tb_writer = tf.summary.create_file_writer(logdir)

    return tb_writer, tb_callback

def write_to_tensorboard(tb_writer, history, outcomes, episode):
    # Add the metrics and graph to TensorBoard
    with tb_writer.as_default():
        for loss in ["value_output_loss", "policy_output_loss"]:
            tf.summary.scalar(name=loss, data=history.history[loss][0], step=episode)
        for acc in ["value_output_accuracy", "policy_output_accuracy"]:
            tf.summary.scalar(name=acc, data=history.history[acc][0], step=episode)

        if "concept_bottleneck_output_loss" in history.history:
            tf.summary.scalar(name="concept_bottleneck_output_loss", data=history.history["concept_bottleneck_output_loss"][0], step=episode)
        if "concept_bottleneck_output_accuracy" in history.history:
            tf.summary.scalar(name="concept_bottleneck_output_accuracy", data=history.history["concept_bottleneck_output_accuracy"][0], step=episode)

        black_wins = (outcomes == 1).sum()
        white_wins = (outcomes == -1).sum()
        
        winrate = black_wins / len(outcomes)

        tf.summary.text("Game results", f"B-W: {black_wins}-{white_wins} -> Winrate black: {round(winrate, 2)}", step=episode)

        # Plot the winrate
        tf.summary.scalar(name="winrate_black", data=winrate, step=episode)

        #tf.summary.trace_on()
        # Call only one tf.function when tracing.
        # Write the graph to a file
        #tf.summary.trace_export("graph", 0)
    
    tb_writer.flush()