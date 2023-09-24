import numpy as np

def normalize(x: np.array):
    """
    Normalize the distribution
    """
    return x / np.sum(x)