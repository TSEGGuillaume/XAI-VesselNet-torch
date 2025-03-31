import numpy as np

def compute_fisher_contrast_noise_ratio(signal: np.ndarray, noise: np.ndarray, abs: bool = True) -> np.ndarray:
    """ Compute the Fisher's contrast-to-noise ratio.

    Args
    ----------
        signal : np.ndarray
            the signal distribution
        noise : np.ndarray
            the noise distribution
        abs : bool
            whether to use the absolute value of distributions

    Returns
    ----------
        The Fisher's contrast-to-noise ratio
    """
    if abs:
        signal = np.abs(signal)
        noise = np.abs(noise)

    num = np.mean(signal) - np.mean(noise)
    denom = np.var(signal) + np.var(noise)

    return np.square(num) / denom
