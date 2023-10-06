import os
from utils import config

def folder_setup():
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