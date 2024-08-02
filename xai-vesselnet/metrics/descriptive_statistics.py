import numpy as np
from scipy.stats import skew as skewness

def stat_distribution_global(distribution):
    """
    Compute global statistics from the distributions.
    Computed statistics : sum, negative values count, positive values count

    Args:
        distribution : The distribution

    Returns:
        The dictionnary of the computed statistics.
    """
    return {
        "sum"       : np.sum(distribution),
        "nonzero"   : [np.sum(distribution < 0), np.sum(distribution > 0)]
    }


def stat_distribution_tendancy(distribution):
    """
    Describe the tendency of a distribution of a single variable (descriptive statistics)
    Computed statistics : mean, median

    Args:
        distribution : The distribution

    Returns:
        The dictionnary of the computed statistics.
    """
    return {
        "mean"      : np.mean(distribution),
        "median"    : np.median(distribution)
    }


def stat_distribution_variability(distribution):
    """
    Describe the variability of a distribution of a single variable (descriptive statistics)
    Computed statistics : minimal value, maximal value, standard deviation

    Args:
        distribution : The distribution

    Returns:
        The dictionnary of the computed statistics.
    """
    return {
        "range"     : [np.min(distribution), np.max(distribution)],
        "std_dev"   : np.std(distribution)
    }


def stat_distribution_quantile(distribution):
    """
    Compute various quantiles of a distribution
    Computed quantiles : quartile, 99% percentile

    Args:
        distribution : The distribution

    Returns:
        The dictionnary of the computed quantiles.
    """
    return {
        "quartile"      : np.quantile(distribution, [0.25, 0.75]).tolist(),
        "percentile_99" : np.percentile(distribution, [1, 99]).tolist()
    }


def stat_distribution_shape(distribution):
    """
    Describe the shape of a distribution.
    Computed descriptors : skewness

    Args:
        distribution : The distribution

    Returns:
        The dictionnary of the computed descriptors.
    """
    return {
        "skewness" : skewness(distribution)
    }


def univariate_analysis(distribution):
    """
    Compute various statistics of a single variable
    See functions:
      - stat_distribution_global,
      - stat_distribution_tendancy,
      - stat_distribution_variability,
      - stat_distribution_quantile, 
      - stat_distribution_shape

    Args:
        distribution : The distribution

    Returns:
        The dictionnary of the computed statistics.
    """
    res_descriptors = {}

    descriptors = [
        stat_distribution_global, 
        stat_distribution_tendancy,
        stat_distribution_variability,
        stat_distribution_quantile,
        stat_distribution_shape,
    ]

    for desc in descriptors:
        res_descriptors = res_descriptors | desc(distribution)

    return res_descriptors