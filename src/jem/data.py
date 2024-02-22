import numpy as np
from concepts import generate_static_concept_datasets
from .concepts import *
from typing import Tuple, List

def get_explanation_list():
    return [
        'a generic move not tied to a strategy',
        'creates one eye',
        'creates two eyes',
        'provides center dominance',
        'provides area advantage',
    ]

def init_confusion_matrix():
    # Initialize the confusion matrix
    return {
        "null": {
            "null": 0,
            "one_eye": 0,
            "two_eyes": 0,
            "center_dominance": 0,
            "area_advantage": 0,
        },
        "one_eye": {
            "null": 0,
            "one_eye": 0,
            "two_eyes": 0,
            "center_dominance": 0,
            "area_advantage": 0,
        },
        "two_eyes": {
            "null": 0,
            "one_eye": 0,
            "two_eyes": 0,
            "center_dominance": 0,
            "area_advantage": 0,
        },
        "center_dominance": {
            "null": 0,
            "one_eye": 0,
            "two_eyes": 0,
            "center_dominance": 0,
            "area_advantage": 0,
        },
        "area_advantage": {
            "null": 0,
            "one_eye": 0,
            "two_eyes": 0,
            "center_dominance": 0,
            "area_advantage": 0,
        },
    }


def get_data(agents, cases_to_sample, board_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """
    Load the data
    """

    concepts_functions = [
        one_eye,
        two_eyes,
        center_dominance,
        area_advantage,
    ]

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

    for concept_function in concepts_functions:
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


def translate_explanation(explanation: str):
    """
    Translate the explination
    """
    if explanation == 'a generic move not tied to a strategy':
        return "null"
    elif explanation == 'creates one eye':
        return "one_eye"
    elif explanation == 'creates two eyes':
        return "two_eyes"
    elif explanation == 'provides center dominance':
        return "center_dominance"
    elif explanation == 'provides area advantage':
        return "area_advantage"
    else:
        raise ValueError("Invalid explination: " + explanation)
