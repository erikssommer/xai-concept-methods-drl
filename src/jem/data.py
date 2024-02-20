import numpy as np
from concepts import generate_static_concept_datasets
from .concepts import *
from typing import Tuple, List
from env import gogame

def get_data(agents, cases_to_sample, board_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """
    Load the data
    """
    explinations = [
        'a generic move not tied to a strategy',
        'creates one eye',
        'creates two eyes',
        'provides center dominance',
        'provides area advantage',
        #'leads to a win'
    ]

    concepts_functions = [
        one_eye,
        two_eyes,
        center_dominance,
        area_advantage,
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
    explinations = np.array([explination + [0] * (max_len - len(explination)) for explination in explinations])

    all_positive_cases = []
    all_negative_cases = []
    all_explinations = []
    all_labels = []

    for concept_function in concepts_functions:
        # Get the concept explination
        concept_explination = concept_function()
        integer_format = convert_explination_to_integers(concept_explination, vocab, max_len)
        positive_cases, _ = generate_static_concept_datasets(cases_to_sample, agents, board_size, concept_function, nn_format=True)

        all_positive_cases.extend(positive_cases)
        all_labels.extend([0] * len(positive_cases))
        all_explinations.extend([integer_format] * len(positive_cases))

        # For each positive state, create a negative case with all the other explinations
        for positive_case in positive_cases:
            for _, explination in enumerate(explinations):
                if not np.array_equal(explination, integer_format):
                    all_negative_cases.append(positive_case)
                    all_labels.append(1)
                    all_explinations.append(explination)

    all_states = np.array(all_positive_cases + all_negative_cases)
    all_explinations = np.array(all_explinations, dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.float32)

    return all_states, all_explinations, all_labels, max_len, vocab

def convert_integers_to_explinations(integers: np.ndarray, vocab: dict) -> List[str]:
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

def convert_explination_to_integers(explination: str, vocab: dict, max_len: int) -> np.ndarray:
    """
    Convert the explination to integers
    """
    explination = explination.split()
    explination = [vocab[word] for word in explination]
    explination = np.array(explination, dtype=np.int32)

    # Pad the explination
    explination = np.pad(explination, (0, max_len - len(explination)), 'constant', constant_values=0)

    return explination


def translate_explination(explination: str):
    """
    Translate the explination
    """
    if explination == 'a generic move not tied to a strategy':
        return "null"
    elif explination == 'creates one eye':
        return "eye"
    elif explination == 'creates two eyes':
        return "double_eye"
    elif explination == 'provides center dominance':
        return "center_dominance"
    elif explination == 'provides area advantage':
        return "area_advantage"
    elif explination == 'leads to a win':
        return "win"
    else:
        raise ValueError("Invalid explination: " + explination)