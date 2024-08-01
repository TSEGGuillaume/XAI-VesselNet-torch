import logging

from numpy import ndarray
from scipy import ndimage as ndi

logger = logging.getLogger("app")


def compute_vessel_thickness(data: ndarray, landmark_pos: tuple) -> float:
    """
    Compute the vessel thickness (by Exact Euclidean Distance Transform) at a specified position.
    TODO : thickness is given in px. A good thing would be to convert to mm using volume's affine

    Args:
        data            : The image of the vessels.
        landmark_pos    : The position of the observed landmark.

    Returns:
        vessel_thickness : The thickness of the vessel at the specified position.
    """
    dist_map = distance_map(data, method="edt")

    # EEDT gives the minimal RADIUS to the background, double for the diameter
    vessel_thickness = dist_map[landmark_pos] * 2

    logger.debug("Vessel diameter : {} (vx)".format(vessel_thickness))

    return vessel_thickness


def distance_map(I: ndarray, method: str = "edt") -> ndarray:
    """
    Distance map transformation. Compute the distance map from a binary image.
    Implemented distance methods :
        - "edt" : Exact Euclidean Distance Transform

    Parameters
        I       : The image to transform
        method  : The key of the transformation to compute ; see implemented distance methods

    Returns
        The distance map
    """

    # Exact Euclidean Distance Transform
    if method == "edt":
        distance_map = ndi.distance_transform_edt(I)
    else:
        raise NotImplementedError(
            "Selected method not available. Available methods : `edt`"
        )

    return distance_map
