import math
import numpy as np

def distance(p1: tuple | list, p2: tuple | list, norm: str = "L2"):
    """
    Compute the distance between two n-D points.
    Implemented distance methods:
        - "L2" : Euclidean distance

    Args:
        p1 (tuple)  : The first point
        p2 (tuple)  : The second point. Must be the same dimension than p1.
        norm (str)  : The name of the distance method. See Implemented distance methods.

    Returns:
        The distance according to the specified method (float).
    """
    assert len(p1) == len(p2), "Point must have the same dimensions."  # Check the dimensions of the points

    if norm == "L2":
        return math.sqrt(np.sum(np.square(np.subtract(p1, p2))))
    
    else:
        raise NotImplementedError
