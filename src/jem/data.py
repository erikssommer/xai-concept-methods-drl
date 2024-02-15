import numpy as np
from concepts.static_concepts import *

def get_data():
    """
    Load the data
    """
    explinations = [
        'a generic move not tied to a strategy',
        'creates an eye',
        'creates an double eye',
        'provides center dominance',
        'gives area advantage',
        'leads to a win'
    ]

    # Apply one hot encoding to the explinations
    vocab = {}
    vocab[''] = 0
    for explination in explinations:
        for word in explination.split():
            if word not in vocab:
                vocab[word] = len(vocab)
            
    explinations = [[vocab[word] for word in explination.split()] for explination in explinations]

    # Pad the explinations
    max_len = max([len(explination) for explination in explinations])
    explinations = [explination + [0] * (max_len - len(explination)) for explination in explinations]

    print(vocab)
    print(explinations)

    states = []

    # Create the states
    for state in range(len(explinations)):
        states.append(np.random.rand(5, 7, 7))

    # Create the labels
    labels = [0, 0, 1, 0, 1, 0]

    states = np.array(states)
    explinations = np.array(explinations)
    labels = np.array(labels, dtype=np.float32)

    return states, explinations, labels, vocab

def convert_integers_to_explinations(integers: np.ndarray, vocab: dict):
    """
    Convert the integers to explinations
    """
    explinations = []

    for integer in integers:
        explination = " ".join([word for word, index in vocab.items() if index == integer and index != 0])
        explinations.append(explination)

    # Make it a string
    explinations = " ".join(explinations)

    # Strip the last space
    explinations = explinations.strip()
    
    short_hand = translate_explination(explinations)

    return short_hand

    

def init_confusion_matrix():
    # Initialize the confusion matrix
    confusion_matrix = {
        "null": {
            "null": 0,
            "eye": 0,
            "double_eye": 0,
            "center_dominance": 0,
            "area_advantage": 0,
            "win": 0
        },
        "eye": {
            "null": 0,
            "eye": 0,
            "double_eye": 0,
            "center_dominance": 0,
            "area_advantage": 0,
            "win": 0
        },
        "double_eye": {
            "null": 0,
            "eye": 0,
            "double_eye": 0,
            "center_dominance": 0,
            "area_advantage": 0,
            "win": 0
        },
        "center_dominance": {
            "null": 0,
            "eye": 0,
            "double_eye": 0,
            "center_dominance": 0,
            "area_advantage": 0,
            "win": 0
        },
        "area_advantage": {
            "null": 0,
            "eye": 0,
            "double_eye": 0,
            "center_dominance": 0,
            "area_advantage": 0,
            "win": 0
        },
        "win": {
            "null": 0,
            "eye": 0,
            "double_eye": 0,
            "center_dominance": 0,
            "area_advantage": 0,
            "win": 0
        }
    }

    return confusion_matrix


def translate_explination(explination: str):
    """
    Translate the explination
    """
    if explination == 'a generic move not tied to a strategy':
        return "null"
    elif explination == 'creates an eye':
        return "eye"
    elif explination == 'creates an double eye':
        return "double_eye"
    elif explination == 'provides center dominance':
        return "center_dominance"
    elif explination == 'gives area advantage':
        return "area_advantage"
    elif explination == 'leads to a win':
        return "win"
    else:
        raise ValueError("Invalid explination: " + explination)