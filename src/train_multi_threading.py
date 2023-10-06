from rl import rl_multiprocessing
from utils import Timer, folder_setup
import os
import logging

def train_models():
    # Start a timer
    timer = Timer()
    timer.start_timer()

    # Train the models
    rl_multiprocessing()

    # End the timer
    timer.end_timer()


def set_logging_level():
    # Create the folder containing the models if it doesn't exist
    if not os.path.exists('../log'):
        os.makedirs('../log')
    
    # Set the logging level
    logging.basicConfig(filename='../log/logfile.log', 
                        level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)s %(name)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    

# Main method
if __name__ == "__main__":
    set_logging_level()
    folder_setup()
    train_models()
        