import numpy as np
from concepts import generate_static_concept_datasets
from .concepts import *
from typing import Tuple, List
import pickle
import os

def get_explanation_list():
    concept_list = []
    for concept_function in concept_functions_to_use():
        # Returns the concept explination if no board state is given
        concept_list.append(concept_function())

    return concept_list

def init_confusion_matrix():
    # Create a confusion matrix dictionary from the name of the concepts
    confusion_matrix = {}
    for concept_function in concept_functions_to_use():
        confusion_matrix[concept_function.__name__] = {}
        for c_f in concept_functions_to_use():
            confusion_matrix[concept_function.__name__][c_f.__name__] = 0

    return confusion_matrix

def translate_explanation(explanation: str):
    """
    Translate the explination
    """
    for concept_function in concept_functions_to_use():
        if explanation == concept_function():
            return concept_function.__name__

def generate_data(agents, cases_to_sample, board_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """
    Load the data
    """

    print(f'Generating dataset for board size {board_size}')

    # Apply one hot encoding to the explinations
    explanations = get_explanation_list()
    vocab = {}
    vocab[''] = 0
    for explanation in explanations:
        for word in explanation.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    explanations = [[vocab[word] for word in explanation.split()]
                    for explanation in explanations]

    # Pad the explinations
    max_len = max([len(explanation) for explanation in explanations])
    explanations = np.array(
        [explination + [0] * (max_len - len(explination)) for explination in explanations])

    all_positive_cases = []
    all_negative_cases = []
    all_explanations = []
    all_labels = []

    for concept_function in concept_functions_to_use():
        # Get the concept explination
        concept_explanation = concept_function()
        integer_format = convert_explanation_to_integers(
            concept_explanation, vocab, max_len)
        positive_cases, _ = generate_static_concept_datasets(
            cases_to_sample, agents, board_size, concept_function, sample_ratio=1, nn_format=True)

        all_positive_cases.extend(positive_cases)
        all_labels.extend([0] * len(positive_cases))
        all_explanations.extend([integer_format] * len(positive_cases))

        # For each positive state, create a negative case with all the other explinations
        for positive_case in positive_cases:
            for _, explanation in enumerate(explanations):
                if not np.array_equal(explanation, integer_format):
                    all_negative_cases.append(positive_case)
                    all_labels.append(1)
                    all_explanations.append(explanation)

    all_states = np.array(all_positive_cases + all_negative_cases)
    all_explanations = np.array(all_explanations, dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.float32)

    return all_states, all_explanations, all_labels, max_len, vocab


def convert_integers_to_explanations(integers: np.ndarray, vocab: dict) -> List[str]:
    """
    Convert the integers to explinations
    """
    explanations = []

    for integer in integers:
        explanation = " ".join(
            [word for word, index in vocab.items() if index == integer and index != 0])
        explanations.append(explanation)

    # Make it a string
    explanations = " ".join(explanations)

    # Strip the last space
    explanations = explanations.strip()

    short_hand = translate_explanation(explanations)

    return short_hand


def convert_explanation_to_integers(explanation: str, vocab: dict, max_len: int) -> np.ndarray:
    """
    Convert the explination to integers
    """
    explanation = explanation.split()
    explanation = [vocab[word] for word in explanation]
    explanation = np.array(explanation, dtype=np.int32)

    # Pad the explination
    explanation = np.pad(
        explanation, (0, max_len - len(explanation)), 'constant', constant_values=0)

    return explanation

def load_datasets_from_pickle(board_size):
    # Test if the folder exists
    if not os.path.exists(f'../datasets/jem/board_size_{board_size}'):
        raise FileNotFoundError(f'No dataset found for board size {board_size}')
    
    print(f'Loading dataset for board size {board_size}')

    with open(f'../datasets/jem/board_size_{board_size}/states.pkl', 'rb') as f:
        states = pickle.load(f)
    with open(f'../datasets/jem/board_size_{board_size}/explanations.pkl', 'rb') as f:
        explanations = pickle.load(f)
    with open(f'../datasets/jem/board_size_{board_size}/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(f'../datasets/jem/board_size_{board_size}/max_sent_len.pkl', 'rb') as f:
        max_sent_len = pickle.load(f)
    with open(f'../datasets/jem/board_size_{board_size}/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    return states, explanations, labels, max_sent_len, vocab

def save_datasets_to_pickle(states, explanations, labels, max_sent_len, vocab, board_size):
    # Create the directories if they don't exist
    os.makedirs(f'../datasets', exist_ok=True)
    os.makedirs(f'../datasets/jem', exist_ok=True)
    os.makedirs(f'../datasets/jem/board_size_{board_size}', exist_ok=True)

    print(f'Saving dataset for board size {board_size}')

    with open(f'../datasets/jem/board_size_{board_size}/states.pkl', 'wb') as f:
        pickle.dump(states, f)
    with open(f'../datasets/jem/board_size_{board_size}/explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)
    with open(f'../datasets/jem/board_size_{board_size}/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open(f'../datasets/jem/board_size_{board_size}/max_sent_len.pkl', 'wb') as f:
        pickle.dump(max_sent_len, f)
    with open(f'../datasets/jem/board_size_{board_size}/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
