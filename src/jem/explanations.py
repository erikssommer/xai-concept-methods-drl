# Enum class for explanations and their corresponding values

from enum import Enum

class Explanations(Enum):
    """
    Enum class for explanations and their corresponding reward values
    """
    NULL = "a generic move not tied to a strategy", 0
    ONE_EYE = "creates one eye where the opponent cannot place a stone", 0.1
    TWO_EYES = "creates two eyes resulting in forming a living group", 0.5
    CAPTURE_A_STONE = "captures one of the opponent's stones", 0.2
    CAPTURE_GROUP_OF_STONES = "captures a group of the opponent's stones", 0.5
    AREA_ADVANTAGE = "creates an area advantage", 0.2