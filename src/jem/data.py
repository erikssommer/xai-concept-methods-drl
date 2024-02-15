from .model import JointEmbeddingModel
import numpy as np

def get_data(batch_size):
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
    for explination in explinations:
        for word in explination.split():
            if word not in vocab:
                vocab[word] = len(vocab)
            
    explinations = [[vocab[word] for word in explination.split()] for explination in explinations]

    # Pad the explinations
    max_len = max([len(explination) for explination in explinations])
    explinations = [explination + [0] * (max_len - len(explination)) for explination in explinations]

    states = []

    # Create the states
    for state in range(len(explinations)):
        states.append(np.random.rand(4, 7, 7))

    # Create the labels
    labels = [1, 0, 0, 0, 0, 0]

    states = np.array(states)
    explinations = np.array(explinations)
    labels = np.array(labels, dtype=np.float32)

    return states, explinations, labels, len(vocab)