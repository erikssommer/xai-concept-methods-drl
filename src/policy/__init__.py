from .convnet import ConvNet
from .resnet import ResNet
from .basenet import BaseNet
from .fast_predictor import FastPredictor
from .lite_model import LiteModel
from .conceptnet import ConceptNet

def get_policy(model_type, board_size, nr_of_concepts=0, path=None) -> BaseNet:
    """
    Get the policy model
    
    Args:
        model_type (str): The type of model to use
        board_size (int): The size of the board
        path (str): The path to the model
        
    Returns:
        BaseNet: The policy model
    """
    if model_type == "resnet":
        policy = ResNet(board_size, load_path=path)
    elif model_type == "convnet":
        policy = ConvNet(board_size, load_path=path)
    elif model_type == "conceptnet":
        policy = ConceptNet(board_size, nr_of_concepts, load_path=path)
    else:
        # Throw an error if the model type is not recognized
        raise ValueError(f"Model type {model_type} not recognized")
    
    return policy