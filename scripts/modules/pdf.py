import numpy as np

def multivariate_gaussian_pdf(X : np.array, mu : np.array, cov : np.array) -> float:
    """
    Computes the likelihood of a vector w.r.t. the multivariate gaussian PDF.

    Parameters
    ==========
    X: numpy.array.
        The input array.
    mu: numpy.array.
        The mean array.
    cov: numpy.array.
        The covariance array.
    Returns
    ===========
    p: float.
        The likelihood of X given the mean mu and covariance cov.
    """
    diff = X.reshape(-1,1) - mu.reshape(-1,1)
    exp_factor = float( - 0.5 * np.dot( np.dot ( diff.T, np.linalg.inv(cov) ), diff ) )
    norm_factor = float ( np.linalg.det(2 * np.pi * cov) ** (-0.5) )
    p = norm_factor * np.exp( exp_factor )
    return p