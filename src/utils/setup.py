import os
import random
import string
import json

from utils import config

def folder_setup(model_type):
    board_size = config.board_size
    # Create the folder containing the models if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs(f'../models/')
    if not os.path.exists(f'../models/training'):
        os.makedirs(f'../models/training/')
    if not os.path.exists(f'../models/training/{model_type}'):
        os.makedirs(f'../models/training/{model_type}/')
    if not os.path.exists(f'../models/training/{model_type}/board_size_{board_size}'):
        os.makedirs(f'../models/training/{model_type}/board_size_{board_size}/')
    else:
        # Delete the model folders
        folders = os.listdir(f'../models/training/{model_type}/board_size_{board_size}')
        for folder in folders:
            # Test if ends with .keras
            if not folder.endswith('.keras'):
                # Delete the folder even if it's not empty
                os.system(f'rm -rf ../models/training/{model_type}/board_size_{board_size}/{folder}')
            else:
                # Delete the file
                os.remove(f'../models/training/{model_type}/board_size_{board_size}/{folder}')

def concept_folder_setup_and_score(concept_type, model_type, board_name, session_name, concept_name, name, score):
    # Remove the files if they exist
    if os.path.exists(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}/{concept_name}/"):
        # Test if epoch folder exists
        if os.path.exists(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}/{concept_name}/{name}"):
            # Remove all files in the epoch folder
            # Fist save the 
            for file in os.listdir(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}/{concept_name}/{name}"):
                os.remove(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}/{concept_name}/{name}/{file}")

    os.makedirs("../concept_presences", exist_ok=True)
    os.makedirs(f"../concept_presences/{concept_type}", exist_ok=True)
    os.makedirs(f"../concept_presences/{concept_type}/{model_type}", exist_ok=True)
    os.makedirs(f"../concept_presences/{concept_type}/{model_type}/{board_name}", exist_ok=True)
    os.makedirs(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}", exist_ok=True)
    os.makedirs(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}/{concept_name}", exist_ok=True)
    os.makedirs(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}/{concept_name}/{name}", exist_ok=True)

    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    # Save concept presences in json file
    with open(f"../concept_presences/{concept_type}/{model_type}/{board_name}/{session_name}/{concept_name}/{name}/{random_suffix}.json", "w") as f:
        json.dump(score, f)