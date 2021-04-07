import numpy as np

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
    noise_array = np.random.normal(loc=0, scale=odom_variance ** 0.5, size=(n_particles)) 
    noise_odom = odom + noise_array
    particles[:,0] = particles[:,0] + noise_odom
    return

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
    particles[:,2] = p_z_given_S(particles)
    return

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