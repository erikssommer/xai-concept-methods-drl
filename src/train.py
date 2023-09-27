import rl
from utils import Timer
import os
import logging

def train_models():
    model = rl.RL()

    # Start a timer
    timer = Timer()
    timer.start_timer()

    # Train the models
    model.learn()

    # End the timer
    timer.end_timer()
    
def setup():
    # Create the folder containing the models if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs('../models')
    else:
        # Delete the model folders
        folders = os.listdir('../models')
        for folder in folders:
            os.rmdir(f'../models/{folder}')
    
    # Create the folder containing the visualizations if it doesn't exist
    if not os.path.exists('../log/visualization'):
        os.makedirs('../log/visualization')

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
    setup()
    train_models()