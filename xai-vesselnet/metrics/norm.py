import numpy as np 


def compute_norm(distribution: np.ndarray, norm_type: str = "L1") -> np.ndarray:
    """ Compute the norm of the distribution.

    Args
    ----------
        distribution : np.ndarray
            the distribution
        norm_type    : str
            the type of the norm ("L1", "L2", "SQL2"). "L1" by default

    Returns
    ----------
        The norm of the distribution

    Raises
    ----------
        `NotImplementedError` if the provided `norm_type` is not implemented.
    """
    if norm_type == "L1":
        norm_fun = lambda matrix: np.sum(np.absolute(matrix))
    elif norm_type == "L2":
        norm_fun = lambda matrix: np.sqrt(np.sum(np.square(matrix)))
    elif norm_type == "SQL2":
        norm_fun = lambda matrix: np.sum(np.square(matrix))
    else:
        raise NotImplementedError(f"{norm_type} norm type does not exist. Available norms are : `L1`, `L2` or `SQL2`")
    
    return norm_fun(distribution)