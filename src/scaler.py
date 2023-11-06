
import sys
sys.path.append('../')

import numpy as np


def scale_action(action_space, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = action_space.low, action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space, action: np.ndarray) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param action: Action to un-scale
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (action + 1.0) * (high - low))
