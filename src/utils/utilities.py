import numpy as np

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize the distribution
    """
    return x / np.sum(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def plot_distribution(distribution: np.array) -> None:
    """
    Plot the distribution
    """
    import matplotlib.pyplot as plt
    plt.bar(np.arange(len(distribution)), distribution)
    plt.show()