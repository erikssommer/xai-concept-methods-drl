from learning.rl import RL
from utils.timer import Timer
import os

def train_models():
    rl = RL()

    # Start a timer
    timer = Timer()
    timer.start_timer()

    # Train the models
    rl.learn()

    # End the timer
    timer.end_timer()
    
def setup():
    # Create the folder containing the models if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs('../models')
    
    # Create the folder containing the visualizations if it doesn't exist
    if not os.path.exists('./visualization'):
        os.makedirs('./visualization')



# Main method
if __name__ == "__main__":
    setup()
    train_models()