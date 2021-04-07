import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def multivariate_gaussian_pdf(X, mu, cov):
    diff = X.reshape(-1,1) - mu.reshape(-1,1)
    exp_factor = float( - 0.5 * np.dot( np.dot ( diff.T, np.linalg.inv(cov) ), diff ) )
    norm_factor = float ( np.linalg.det(2 * np.pi * cov) ** (-0.5) )
    p = norm_factor * np.exp( exp_factor )
    return p

def measurement_model_landmark(particle_xy : np.array, landmark_xy : np.array, covariance : np.array, sensitivity : float, fpr : float, epsilon : float = 20) -> float :
    """
    Parameters
    =========
    particle_xy : np.array.
        The 1D array of the (x,y) coordinates of a particle.
    landmark_xy : np.array.
        The 1D array of the (x,y) coordinates of a landmark.
    covariance : np.array.
        The 2D array of the covariance matrix of the measurement model.

    Returns
    =========
    p: float.
        The probability of the landmark being measured by the particle provided.
    """
    norm = 1./(sensitivity + fpr)
    diff = ( landmark_xy.reshape(2,1) - particle_xy.reshape(2,1) )
    distance = np.linalg.norm(diff)
    g_pdf = multivariate_gaussian_pdf(landmark_xy, particle_xy, covariance)
    p_match = None
    if(distance <= epsilon):
        p_match = sensitivity#norm * sensitivity * g_pdf
    else:
        p_match = 0.0 #norm * fpr * g_pdf
    return p_match

def measurement_model_landmark_batch(particle_xy_array : np.array, landmark_xy : np.array, covariance : np.array) -> np.array:
    """
    TODO: NOT COMPLETED
    Parameters
    =========
    particle_xy : np.array.
        The 2D array (n_particles, 2) of the (x,y) coordinates for each particle.
    landmark_xy : np.array.
        The 1D array of the (x,y) coordinates of a landmark.
    covariance : np.array.
        The 2D array of the covariance matrix of the measurement model.

    Returns
    =========
    p: np.array.
        The 1D array of the likelihoods of the landmarks being measured by the provided particle.
    """
    diff_array = ( landmark_xy.reshape(1,2) - particle_xy.reshape(-1,2) ).T # (2, n_particles): each column corresponds to a particle
    # TODO
    return

def match(query_features : np.array, dataset_features : np.array) -> int:
    """
    Finds the best correspondence to the query input on the dataset.

    Parameters
    =======
    query_features: numpy.array.
        The 1D array (n_features,) of features from the query.
    dataset_features: numpy.array.
        The 2D array (size_dataset, n_features) of the features from the dataset.
    
    Returns
    ==========
    match_id: int.
        The index of the dataset feature that maximizes the similarity score.
    match_similarity: float.
        The similarity between the query and the matched features.
    """
    q = query_features.reshape(1,-1)
    D = dataset_features
    s = cosine_similarity(D,q).flatten()
    match_id = np.argmax(s)
    match_similarity = float(s[match_id])
    return match_id, match_similarity

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    p_xy = np.array([1.,0.])
    l_xy = np.array([0.,0.])
    Q = 10 * np.array([[0.05, 0.01], [0.01, 0.05]])
    
    xs = np.linspace(-5, 5, 100)
    ys = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(xs,ys)
    zs = np.zeros_like(xx)
    for i in range(zs.shape[0]):
        for j in range(zs.shape[1]):
            l_xy = np.array([xx[i,j], yy[i,j]])
            zs[i,j] = measurement_model_landmark(p_xy, l_xy, Q)
    _ = plt.contourf(xs,ys,zs)
    plt.show()
    exit(0)