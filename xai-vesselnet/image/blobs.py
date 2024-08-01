import logging

import numpy as np
from numpy import ndarray

from skimage.filters import frangi, threshold_otsu
from skimage.measure import label, regionprops


logger = logging.getLogger("app")


def detect_blob(
    I: ndarray,
    sigma_min: float = 2.0,
    sigma_max: float = 16.0,
    N_sigma: int = 14,
    threshold: float = None,
) -> ndarray:
    """
    Search for blobs in an image.
    We use Frangi's algorithm to produce a blobness-filtered image, followed by a threshold that output a binary blobs mask.

    Args:
        I           : The image (H,W,[D]).
        sigma_min   : The minimum deviation for Gaussian scale-space. 2.0 by default.
        sigma_max   : The maximum deviation for Gaussian scale-space. 16.0 by default.
        N_sigma     : The number of sigmas to use between [sigma_min, sigma_max]. 14 by default.
        threshold   : The blobness threshold value. If `None` (default), Otsu will be used.

    Note: sigma_min and sigma_max are included in the sigmas.

    Returns:
        blobs (ndarray) : The blobs mask.
    """
    sigma_step = (sigma_max - sigma_min) / N_sigma
    logger.info(
        "Search for blobs : ({}, {}) | {} steps.".format(
            sigma_min, sigma_max, sigma_step
        )
    )

    frangi_beta = 0.95  # Sensitivity to deviation from a blob-like structure
    frangi_alpha = (
        1 - frangi_beta
    )  # Sensitivity to deviation from a plate-like structure
    final_sigma_max = (
        sigma_max + sigma_step
    )  # To include the provided sigma_max value in the scale-space

    I_blob = frangi(
        I,
        sigmas=np.arange(sigma_min, final_sigma_max, sigma_step),
        alpha=frangi_alpha,
        beta=frangi_beta,
        black_ridges=False,
        mode="constant",
        cval=0,
    )

    if threshold == None:
        # The choice of nbins is debatable, but we chose nbins=256 because our intuition about blob detection was initiated by visual observations of attribution maps,
        # i.e. on intensity-scaled grayscale images of 256 intensity values.
        threshold = threshold_otsu(image=I_blob, nbins=256)

    I_blob = (I_blob > threshold).astype(np.ubyte)

    return I_blob


def compute_blobs_properties(I: ndarray) -> list:
    """
    Labelized an image and measure various properties of the connected components.
    See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

    Args:
        I : The image (H,W,[D])

    Returns:
        props (list) : The list of RegionProperties
    """
    labeled_blobs, nlabels = label(I, connectivity=None, return_num=True)

    logger.info(f"{nlabels} blobs detected")

    if nlabels == 0:
        props = []
        logger.debug("No blob detected, returns empty region properties.")
    else:
        # For debugging
        for lbl_idx in range(1, nlabels + 1):
            logger.debug(f" - Blob {lbl_idx}: {np.sum(labeled_blobs==lbl_idx)}")

        props = regionprops(labeled_blobs)

    return props
