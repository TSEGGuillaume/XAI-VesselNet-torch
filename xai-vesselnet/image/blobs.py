import logging

import numpy as np
from numpy import ndarray

from skimage.filters import frangi, threshold_otsu, threshold_li, threshold_yen
from skimage.measure import label, regionprops_table


logger = logging.getLogger("app")


def detect_blobs(
    I: ndarray,
    sigmas: list[float],
    alpha: float,
    beta: float,
    black_ridges: bool = False,
    threshold: str|float = None,
) -> ndarray:
    """
    Search for blobs in an image.
    We use Frangi's algorithm to produce a blobness-filtered image, followed by a threshold that output a binary blobs mask.
    
    Args:
        I           : The image (H,W,[D]).
        sigmas      : The Gaussian scale-space.
        alpha       : Frangi correction constant that adjusts the filter’s sensitivity to deviation from a plate-like structure.
        beta        : Frangi correction constant that adjusts the filter’s sensitivity to deviation from a blob-like structure.
        black_ridges: When True (the default), the filter detects black ridges; when False, it detects white ridges.
        threshold   : The blobness threshold value. If `None` (default), no threshold is applied.

    Returns:
        blobs (ndarray) : The blobs mask.
    """
    I_blobs = frangi(
        I,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        black_ridges=black_ridges,
        mode="constant",
        cval=0,
    )

    if threshold == None:
        return I_blobs

    elif isinstance(threshold, str):
        if threshold == "otsu":
            # The choice of nbins is debatable, but we chose nbins=256 because our intuition about blob detection was initiated by visual observations of attribution maps,
            # i.e. on intensity-scaled grayscale images of 256 intensity values.
            threshold = threshold_otsu(image=I_blobs, nbins=256)
        elif threshold == "li":
            threshold = threshold_li(image=I_blobs)
        elif threshold == "yen":
            threshold = threshold_yen(image=I_blobs, nbins=256)
        else:
            raise NotImplementedError(f"{threshold} thresholding if not available.")

    I_blobs = (I_blobs > threshold).astype(np.ubyte)

    return I_blobs


def detect_bright_and_dark_blobs(
    I: ndarray,
    sigma_min: float = 1.0,
    sigma_max: float = 6.0,
    N_sigma: int = 5,
    threshold: str|float = None,
) -> ndarray:
    """
    Search for bright and dark blobs in an image.
    We use Frangi's algorithm to produce a blobness-filtered image, followed by a threshold that outputs a binary blobs mask.
    This function detects both bright and dark blobs by filtering the absolute intensity of the image rather than searching for black and white ridges separately.

    Args:
        I           : The image (H,W,[D]).
        sigma_min   : The minimum deviation for Gaussian scale-space. 1.0 by default.
        sigma_max   : The maximum deviation for Gaussian scale-space. 6.0 by default.
        N_sigma     : The number of sigmas to use between [sigma_min, sigma_max[. 5 by default.
        threshold   : The blobness threshold value. If `None` (default), no threshold is applied and Frangi's output is returned.

    Note: sigma_min and sigma_max are included in the sigmas. Hence, we have (N_sigma+1) scales.

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
        0.5
    )  # Sensitivity to deviation from a plate-like structure
    final_sigma_max = (
        sigma_max + sigma_step
    )  # To include the provided sigma_max value in the scale-space

    I = np.abs(I)  # No need to discriminate bright and dark blobs ; we use absolute values

    I_blobs = detect_blobs(
        I,
        sigmas=np.arange(sigma_min, final_sigma_max, sigma_step),
        alpha=frangi_alpha,
        beta=frangi_beta,
        black_ridges=False,
        threshold=threshold,
    ).astype(np.ubyte)

    # I_blobs_bright = detect_blobs(
    #     I,
    #     sigmas=np.arange(sigma_min, final_sigma_max, sigma_step),
    #     alpha=frangi_alpha,
    #     beta=frangi_beta,
    #     black_ridges=False,
    #     threshold=threshold,
    # )
    # I_blobs_dark = detect_blobs(
    #     I,
    #     sigmas=np.arange(sigma_min, final_sigma_max, sigma_step),
    #     alpha=frangi_alpha,
    #     beta=frangi_beta,
    #     black_ridges=True,
    #     threshold=threshold,
    # )

    # I_blobs = np.logical_or(I_blobs_bright, I_blobs_dark).astype(np.ubyte)

    return I_blobs


def compute_blobs_properties(
    I: ndarray,
    selected_props: list[str],
    is_labeled:bool = False,
    include_background:bool = False
) -> tuple:
    """
    Measure specified properties of connected components in a labeled image
    See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

    Args:
        I             : The image (H,W,[D])
        selected_props: The properties to compute for each labeled blob
        is_labeled    : Indicate if I is already labeled. If False, the function will labelize the image.

    Returns:
        tuple (list[dict], ndarray, int) : The list of properties for each blob, the labeled blobs, and the number of blobs detected.
    """
    if not is_labeled:
        labeled_blobs, nlabels = label(I, connectivity=None, return_num=True)
    else:
        labeled_blobs = I.astype(int)
        nlabels = np.max(labeled_blobs)

    logger.info(f"{nlabels} blobs detected")

    props = []

    if nlabels == 0:
        logger.debug("No blob detected, returns empty region properties.")

    else:
        # For what remains an obscure reason, the program raises exception when calculating the properties of some blobs (see except clause.)
        # This causes the function to fail for all blobs because a single blob has triggered the exception.
        # WORK AROUND: we iterate each blob by hand. Blob crashes no longer leads to full failure.
        #
        # We also add the background (label 0) to the list of blobs to iterate over.
        if include_background:
            start_label_idx = 0
        else:
            start_label_idx = 1

        for lbl_idx in range(start_label_idx, nlabels + 1):

            current_blob = (labeled_blobs == lbl_idx).astype(int)

            # 0 is the background
            if lbl_idx != 0:
                current_blob = current_blob * lbl_idx

            logger.debug(f" - Blob {lbl_idx}: {np.sum(current_blob)}")

            try:
                # Causes exceptions for some small blobs (or flat simplex ?) during convex hull calculation.
                # Except is then executed, recalling regionprops_table without props related to convex hull 
                current_props = regionprops_table(
                    current_blob, properties=selected_props
                )

                # `regionprops_table` returns a dictionary of lists, where keys are the props and values are the corresponding property values for each blob.
                # To match the output of `regionprops`, we convert the dictionary of lists to a list of dictionaries 
                # TODO: post-processing non-scalar props, such as d["k-0"], d["k-1"], d["k-2"] into d[k] = (d["k-0"], d["k-1"], d["k-2"])."
                
                current_props = [
                    dict(zip(current_props, t)) for t in zip(*current_props.values())
                ]  # Change the dict of list to list of dict
                
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
                        dict(zip(current_props, t)) for t in zip(*current_props.values())
                    ]  # Change the dict of list to list of dict

                    for elem in intersect:
                        current_props[-1][elem] = -1.0

            # If background "blob"
            if lbl_idx == 0:
                current_props[-1]["label"] = lbl_idx # Fix de label
                current_props[-1]["background"] = True # Specify background
            else:
                current_props[-1]["background"] = False 

            props += current_props

    return props, labeled_blobs, nlabels
