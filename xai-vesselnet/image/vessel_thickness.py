import logging

from numpy import ndarray
from scipy import ndimage as ndi

logger = logging.getLogger("app")


def compute_vessel_thickness(data: ndarray, landmark_pos: tuple|list) -> float:
    """ Compute the vessel thickness.
    
    Description
    ----------
    Compute the vessel thickness using Exact Euclidean Distance Transform and returns the thickness at a specified position.

    Args
    ----------
    data : np.ndarray
        the image (H,W,[D]) of the vessels.

    landmark_pos : tuple|list
        the position to determine the vessel thickness
    
    Returns
    ----------
    out: float
        the vessel thickness at the specified location
    """

    dist_map = distance_map(data, method="edt")

    vessel_thickness = dist_map[landmark_pos] * 2

    logger.debug("Vessel diameter : {} (vx)".format(vessel_thickness))

    return vessel_thickness


def distance_map(I: ndarray, sampling: float|tuple|list = None, method: str = "edt") -> ndarray:
    """ Compute the distance map.
    
    Description
    ----------
    Compute the distance map from a binary image.

    Args
    ----------
    I : np.ndarray
        the image (H,W,[D]) to transform.

    sampling : float|tuple|list
        Spacing of elements along each dimension. Default to `None`, a regular grid of unity will be used.

    method : str
        the key of the transformation to compute ; see implemented distance methods below.
    
    Notes
    ----------
    Implemented distance transform methods :
    - "edt" : Exact Euclidean Distance Transform

    Returns
    ----------
    out: float
        the vessel thickness at the specified location
    
    Raises
    ----------
    `NotImplementedError` if the provided method is not implemented. See implemented distance methods above.
    """
    _l_methods = ["edt"]

    # Exact Euclidean Distance Transform "edt"
    if method == _l_methods[0]:
        distance_map = ndi.distance_transform_edt(I, sampling=sampling)
    else:
        raise NotImplementedError(
            f"Selected method not available. Available methods : {_l_methods}"
        )

    return distance_map
