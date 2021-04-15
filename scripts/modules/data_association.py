import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from modules.pdf import multivariate_gaussian_pdf

def measurement_model_segment_feature(x : float, ways : pd.DataFrame, speed_limit : int, sensitivity : float, fpr : float) -> float:
    """
    Parameters
    ===========
    x: float.
        The accumulated offset of the agent
    ways: pandas.DataFrame.
        The ways of the route where the agent is contained.
    speed_limit: int.
        The measured speed limit.
    sensitivity: float.
        The sensitivity of the sign detection module.
    fpr: float.
        The false positive rate of the detection module.

    Returns
    ==========
    prob: float.
        The probability of detecting the sign at the particle's position.
    """
    # Find map value
    N_ways = len(ways)
    seq_id = 0
    for seq_id in range(N_ways-1):
        c_bef = ways.at[seq_id, "cumulative_length"] 
        c_aft = ways.at[seq_id+1, "cumulative_length"] 
        if x >= c_bef and x < c_aft:
            break
    true_speed_limit = ways.at[seq_id, "speed_limit"]

    # Compute probability
    norm = 1./(sensitivity + fpr)
    prob = 1.0
    if true_speed_limit != speed_limit:
        prob = norm * fpr
    else:
        prob = norm * sensitivity
    return prob

def measurement_model_landmark(particle_xy : np.array, landmark_xy : np.array, radius : float = 20) -> float :
    """
    Parameters
    =========
    particle_xy : np.array.
        The 1D array of the (x,y) coordinates of a particle.
    landmark_xy : np.array.
        The 1D array of the (x,y) coordinates of a landmark.
    radius : np.array.
        The detection range radius.

    Returns
    =========
    p: float.
        The probability of the landmark being measured by the particle provided.
    """
    std = radius/3
    cov = np.diag( [ std ** 2, std ** 2 ] )
    p_match = multivariate_gaussian_pdf(landmark_xy, particle_xy, cov)
    #diff = ( landmark_xy.reshape(2,1) - particle_xy.reshape(2,1) )
    #distance = np.linalg.norm(diff)
    #p_match = 1.0
    #if(distance > radius):
    #    p_match = 0.0
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
    diff_array = ( landmark_xy.reshape(1,2) - particle_xy_array.reshape(-1,2) ).T # (2, n_particles): each column corresponds to a particle
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