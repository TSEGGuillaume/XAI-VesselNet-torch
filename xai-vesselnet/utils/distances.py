def distance(p1:tuple, p2:tuple, norm:str = "L2"):
    """
    Compute the distance between two n-D points.
    Implemented distance methods :
        - "L2" : Euclidean distance 

    Parameters
        p1 (tuple)  : The first point
        p2 (tuple)  : The second point. Must be the same dimension than p1.
        norm (str)  : The name of the distance method. See Implemented distance methods.

    Returns
        The distance according to the specified method (float).
    """
    import math

    assert len(p1) == len(p2) # Check the dimension of the points

    if norm == "L2":
        sum = 0
        for x1, x2 in zip(p1, p2):
            sum += (x1-x2)**2
        return math.sqrt(sum)
    else:
        raise NotImplementedError