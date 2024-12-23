import numpy as np
from numpy import ndarray

def image_total_variation(image:ndarray, neighborhood:str="N6", norm:str="L1", mask=None) -> float:
    """
    Compute the total variation of a 3D image, using L1 norm.
    Note that this algorithm does not take into account image borders. Calculation starts at position corresponding to the radius of the structuring element.
    TODO: implement border management using padding  

    Args:
        image                       : The image. Expected shape is [H,W,D].
        neighborhood (str, optional): The neighborhood to use ("N6" or "N26"). Defaults to "N6".
        norm (str, optional)        : The norm to use ("L1", "L2", "SQL2"). Defaults to "L1".
        mask (ndarray, optional)    : The mask on which to compute the total variation. Same shape as image. If None, entire image is considered. Defaults to None.

    Raises:
        AssertionError: if the image shape is not [H,W,D].
        ValueError: if the selected neighborhood is not available.
        ValueError: if the selected norm is not available.

    Returns:
        float: the total variation TV
    """
    assert len(image.shape) == 3, f"Expected shape for image (H,W,D), got ({image.shape})."

    if mask is None:
        mask = np.ones_like(image) # Consider every voxels

    SE_shape = (3,3,3)

    if neighborhood == "N6":
        SE = np.zeros(SE_shape) # Cross structuring element
        SE[:,1,1] = 1
        SE[1,:,1] = 1
        SE[1,1,:] = 1
    elif neighborhood == "N26": # Cube structuring element
        SE = np.ones(SE_shape)
    else:
        raise ValueError(f"{neighborhood} does not exist. Available neighborhood are : `N6` or `N26`")

    # Plan to implement other norm types
    if norm == "L1":
        norm_fun = lambda matrix: np.sum(np.absolute(matrix))
    elif norm == "L2":
        norm_fun = lambda matrix: np.sqrt(np.sum(np.square(matrix)))
    elif norm == "SQL2":
        norm_fun = lambda matrix: np.sum(np.square(matrix))
    else:
        raise ValueError(f"{norm} norm type does not exist. Available norm are : `L1`, `L2` or `SQL2`")
    
    view = np.lib.stride_tricks.sliding_window_view(image, SE_shape)
    mask_view = np.lib.stride_tricks.sliding_window_view(mask, SE_shape)

    # Flatten the first 3 dimensions to give a list of windows instead of a "grid" of windows
    # (grid_X, grid_Y, grid_Z, SE_shape_X, SE_shape_Y, SE_shape_Z) -> (n_windows, SE_shape_X, SE_shape_Y, SE_shape_Z)
    view = view.reshape( ( np.prod( view.shape[:-len(SE_shape)] ), ) + SE_shape )
    mask_view = mask_view.reshape( ( np.prod( mask_view.shape[:-len(SE_shape)] ), ) + SE_shape )

    sum_variation = 0
    for sub_image, sub_mask in zip(view, mask_view):
        center_val = sub_image[1, 1, 1]
        dynamic_neighbourhood = SE * sub_mask
        sum_variation += norm_fun(dynamic_neighbourhood * center_val - dynamic_neighbourhood * sub_image)

    return sum_variation