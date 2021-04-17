import numpy as np
import pandas as pd
from modules.pdf import multivariate_gaussian_pdf, batch_univariate_gaussian_pdf

def sample_particles(mu : float, sigma : float, n_particles : int , route_idx : int) -> np.array:
    """ 
    Sample particles from a gaussian distribution
    
    Parameters
    ==========
    mu: float.
        Mean of the gaussian distribution.
    sigma: float.
        Standard deviation of the gaussian distribution.
    route_idx: int.
        The route index to be assigned for each particle.

    Returns
    ==========
    particles: numpy.array.
        The (n_particles,3) array, in which column 0 corresponds to the 'x' values, 
        column 1 corresponds to the route indices and column 2 corresponds to the 'w'.
    """
    particles_x = np.random.normal(loc=mu, scale=sigma, size=(n_particles,1))
    particles_r = np.full((n_particles,1), fill_value=route_idx ,dtype="uint32")
    particles_w = np.ones((n_particles,1),dtype=float)
    particles = np.hstack((particles_x, particles_r, particles_w))
    return particles

def motion_model(particles : np.array, odom : float, odom_variance : float):
    """
    Sample from the motion model (prediction). Changes happen inplace.

    Parameters
    ==========
    particles: numpy.array.
        The (n_particles,3) array of particles (x, route_idx, weights).
    odom: float.
        The forward odometry provided.
    odom_variance: float.
        The variance of the odometry.
    """
    n_particles = particles.shape[0]

    # Sample the new particles from the motion model
    noise_array = np.random.normal(loc=odom, scale=odom_variance ** 0.5, size=(n_particles))
    old_xs = particles[:,0]
    new_xs = old_xs + noise_array
    particles[:,0] = new_xs

    # Update their weights - they independ from the route and the route remains unchanged.
    #mean_xs = old_xs + odom # The mean is centered at odom + x
    #new_ws = batch_univariate_gaussian_pdf(new_xs, mean_xs, odom_variance)
    #particles[:,2] = new_ws
    return

# TODO
def measurement_model(particles : np.array, p_z_given_S : callable):
    """
    Update the particles' weights given the conditional of the measurement given the particle.

    Parameters
    ==========
    particles: numpy.array.
        The (n_particles,3) array of particles (x, route_idx, weights).
    p_z_given_S: callable.
        The vectorized function that, given the particles (S), yields their probabilities (numpy.array of shape (n_particles,)).
    """
    particles[:,2] = p_z_given_S(particles) * particles[:,2]
    return

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

def measurement_model_gaussian(particle_xy : np.array, measurement : np.array, covariance : np.array) -> float :
    """
    Parameters
    =========
    particle_xy : np.array.
        The 1D array of the (x,y) coordinates of a particle.
    measurement : np.array.
        The 1D array of the (x,y) coordinates of a landmark.
    covariance : np.array.
        The uncertainty of the measurement.

    Returns
    =========
    p: float.
        The probability of the landmark being measured by the particle provided.
    """
    p_gauss = multivariate_gaussian_pdf(measurement, particle_xy, covariance)
    return p_gauss

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

def low_variance_sampler(weights : np.array) -> np.array:
    """
    Parameters
    ======
    weights: numpy.array.
        The weights of the particles. Will be normalized in this function.

    Returns
    ======
    ids: numpy.array.
        The new ids for sampling the particles with replacement proportional to their weights.
    """
    sum_w = np.sum(weights)
    if(sum_w == 0):
        return np.arange(0, weights.size)
    w = weights/ sum_w
    n_particles = w.size
    delta = 1./n_particles
    r_init = np.random.rand() * delta
    ids = np.zeros((n_particles),dtype="int")
    
    i = 0
    cumulative_w = w[0]
    for k in range(n_particles):
        # The next cumulative weight has to be greater than this
        r = r_init + k * delta
        while r > cumulative_w:
            # Increment the cumulative weight: still not enough
            i += 1
            cumulative_w += w[i]
        ids[k] = i
        
    return ids

if __name__ == "__main__":
    weights = np.array([0.11, 0.12, 0.11, 0.2, 0.5, 1.5])
    ids = low_variance_sampler(weights)
    print(ids)
    exit(0)