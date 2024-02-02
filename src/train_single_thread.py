from rl import rl
from utils import Timer, folder_setup
    
# Main method
if __name__ == "__main__":
    folder_setup()
    
    # Start a timer
    timer = Timer()
    timer.start_timer()

    rl()

    # End the timer
    timer.end_timer()