import rl
from utils import Timer
import os
import logging
from utils import config

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
    board_size = config.board_size
    # Create the folder containing the models if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs(f'../models/')
    if not os.path.exists(f'../models/board_size_{board_size}'):
        os.makedirs(f'../models/board_size_{board_size}/')
    else:
        # Delete the model folders
        folders = os.listdir(f'../models/board_size_{board_size}')
        for folder in folders:
            # Test if ends with .keras
            if not folder.endswith('.keras'):
                # Delete the folder even if it's not empty
                os.system(f'rm -rf ../models/board_size_{board_size}/{folder}')
            else:
                # Delete the file
                os.remove(f'../models/board_size_{board_size}/{folder}')
    
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