import logging

import numpy as np
from numpy import ndarray

from skimage.filters import frangi, threshold_otsu
from skimage.measure import label, regionprops_table


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

    frangi_beta = 0.5  # Sensitivity to deviation from a blob-like structure
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


def compute_blobs_properties(I: ndarray, selected_props: list[str]) -> list:
    """
    Labelized an image and measure various selected properties of the connected components.
    See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

    Args:
        I : The image (H,W,[D])
        selected_props  : The properties to compute for each labeled blob

    Returns:
        props (list) : The list of RegionProperties
    """
    labeled_blobs, nlabels = label(I, connectivity=None, return_num=True)

    logger.info(f"{nlabels} blobs detected")

    props = []

    if nlabels == 0:
        logger.debug("No blob detected, returns empty region properties.")
    else:
        for lbl_idx in range(1, nlabels + 1):
            logger.debug(f" - Blob {lbl_idx}: {np.sum(labeled_blobs==lbl_idx)}")

            current_blob = (labeled_blobs == lbl_idx).astype(np.ubyte)

            try:
                current_props = regionprops_table(
                    current_blob, properties=selected_props
                )

                # We use `regionprops_table` instead of `regionprops` because `regionprops_table` is executed when called, while `regionprops` is executed when indexed, causing exceptions for some small blobs (or flat simplex ?) during convex hull calculation.
                # `regionprops_table` returns a dictionary of lists, where keys are the props and values are the corresponding property values for each blob.
                # To match the output of `regionprops`, we convert the dictionary of lists to a list of dictionaries #### TODO and perform some post-processing, such as merging d["k-0"], d["k-1"], and d["k-2"] into d[k] = (d["k-0"], d["k-1"], d["k-2"])."

                current_props = [
                    dict(zip(current_props, t)) for t in zip(*current_props.values())
                ]  # Change the dict of list to list of dict
                props += current_props

            except ValueError as e:

                # Assume the problem came from the convex hull
                if e.__str__() == "Surface level must be within volume data range.":
                    problematic_props = [
                        "area_convex",
                        "feret_diameter_max",
                        "image_convex",
                        "solidity",
                    ]

                    # Remove from selected_props the problematic props
                    intersect = set(problematic_props) & set(selected_props)

                    corrected_props = selected_props.copy()
                    for elem in intersect:
                        corrected_props.remove(elem)

                    current_props = regionprops_table(
                        current_blob, properties=corrected_props
                    )

                    current_props = [
                        dict(zip(current_props, t))
                        for t in zip(*current_props.values())
                    ]  # Change the dict of list to list of dict

                    for elem in intersect:
                        current_props[-1][elem] = -1.0

                    props += current_props

    return props, labeled_blobs, nlabels
