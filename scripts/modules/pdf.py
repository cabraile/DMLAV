import numpy as np

def multivariate_gaussian_pdf(X, mu, cov):
    diff = X.reshape(-1,1) - mu.reshape(-1,1)
    exp_factor = float( - 0.5 * np.dot( np.dot ( diff.T, np.linalg.inv(cov) ), diff ) )
    norm_factor = float ( np.linalg.det(2 * np.pi * cov) ** (-0.5) )
    p = norm_factor * np.exp( exp_factor )
    return p

def batch_univariate_gaussian_pdf(X, mu, var):
    """
    Parameters
    ==========
    X: array-like (n_samples)
        The array of the samples.
    mu: array-like.
        The mean array. If a single mean is provided, then it applies for all X.
        Else, the likelihood is computed for each element in X w.r.t. the mean mu at the same row.
    var: float.
        The variance of the distribution
    """
    diff = X - mu
    diff_sq = diff ** 2.0
    exp_factor = - 0.5 *  (diff_sq / var)
    norm_factor = ( (var ** 0.5) * np.sqrt( 2 * np.pi ) ) ** 0.5
    p = norm_factor * np.exp(exp_factor)
    return p
