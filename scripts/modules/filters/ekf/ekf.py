import numpy as np

class EKF:
    """
    Implements the general Extended Kalman Filter which - in this project - is responsible for
    fusing local and global estimations provided by the proposed methods.
    """

    def __init__(self, prior_mean : np.array, prior_covariance : np.array):
        """
        Parameters
        =================
        prior_mean: numpy.array.
            The initial pose provided.
        prior_covariance: numpy.array
            The initial pose covariance.
        """
        self.mean = prior_mean.reshape(-1,1)
        self.covariance = prior_covariance
        return

    # GETTERS
    # ==============================================

    def get_mean(self) -> np.array:
        """
        Returns
        ==========
        self.mean: numpy.array.
            The current estimation mean.
        """
        return self.mean

    def get_covariance(self) -> np.array:
        """
        Returns
        ==========
        self.covariance: numpy.array.
            The current estimation covariance.
        """
        return self.covariance

    # ==============================================

    # SETTERS
    # ==============================================

    def set_motion_model(self, g: callable):
        """
        Defines the motion model function used by the filter.

        Parameters
        ============
        g: callable.
            The function g(u,x) maps the state x to the next predicted state given the
            action command u. Returns a numpy array of the same shape as the state.
        """
        self.g = g
        return

    def set_jacobian_motion_model(self, G : callable):
        """
        Defines the jacobian of the motion model function used by the filter.

        Parameters
        ============
        G: callable.
            The function G(u,x) is the jacobian of g.
        """
        self.G = G
        return
    
    def set_measurement_model(self, h: callable):
        """
        Defines the measurement model function used by the filter.

        Parameters
        ============
        h: callable.
            The function h(x) which provides the estimated measurement at the state x. 
            Returns a numpy array of the same shape as the measurement.
        """
        self.h = h
        return

    def set_jacobian_measurement_model(self, H : callable):
        """
        Defines the jacobian of the measurement model function used by the filter.

        Parameters
        ============
        h: callable.
            The function H(x) which provides the jacobian of the estimated measurement at the state x.
        """
        self.H = H
        return

    # ==============================================

    # FILTER
    # ==============================================

    def predict(self, u : np.array, u_cov : np.array):
        """
        Performs prediction of the next state's mean and covariance.

        Parameters
        ============
        u: numpy.array.
            The action command (odometry, velocity or any other kind).
        u_cov: numpy.array.
            The covariance of the action command.
        """
        g = self.g
        G = self.G(u, self.mean)
        self.mean = g(u, self.mean)
        self.covariance = np.dot(np.dot(G, self.covariance), G.T) + np.dot(np.dot(G, u_cov), G.T)
        return

    def update(self, z : np.array, Q :np.array) -> np.array:
        """
        Performs correction of the predicted state's mean and covariance.

        Parameters
        ============
        z: numpy.array.
            The measurement of the environment.
        Q: numpy.array.
            The covariance of the measurement model.
        
        Returns
        ============
        K: numpy.array.
            The Kalman gain computed during the current iteration.
        """
        S = self.covariance
        H = self.H(self.mean)
        h = self.h
        I = np.eye(S.shape[0])
        mu = self.mean
        q1 = np.dot( H, np.dot(S, H.T) )
        q2 = q1 + Q
        factor = np.linalg.inv( q2 )
        K = np.dot( np.dot(S, H.T), factor )
        self.mean = mu + np.dot(K, (z - h(mu)))
        self.covariance = np.dot( ( I - np.dot(K, H) ), S )
        return K
 
    # ==============================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def motion_model_odometry_non_linear(u,x):
        c = np.cos(x[2,0])
        s = np.sin(x[2,0])
        g_x = np.array([
            [x[0,0] + c * u[0,0] - s * u[1,0] ],
            [x[1,0] + s * u[0,0] + c * u[1,0]],
            [x[2,0] + u[2,0]]
        ])
        return g_x

    def motion_model_odometry_non_linear_jacobian(u,x):
        c = np.cos(x[2,0])
        s = np.sin(x[2,0])
        G_x = np.array([
            [1., 0., -s * u[0,0] - c * u[1,0] ],
            [0., 1., c * u[0,0] - s * u[1,0]],
            [0., 0., 1.]
        ])
        return G_x

    def motion_model_linear(u,x):
        return u + x
    
    def motion_model_linear_jacobian(u,x):
        return np.eye(x.shape[0])
    
    def measurement_model_linear(x):
        return x
    
    def measurement_model_linear_jacobian(x):
        return np.eye(x.shape[0])

    prior_mean = np.zeros((3,1))
    prior_covariance = np.diag([1e-3,1e-3,1e-4])
    kf = EKF(prior_mean, prior_covariance)
    kf.set_motion_model(motion_model_odometry_non_linear)
    kf.set_jacobian_motion_model(motion_model_odometry_non_linear_jacobian)
    kf.set_measurement_model(measurement_model_linear)
    kf.set_jacobian_measurement_model(measurement_model_linear_jacobian)

    est_position_list = []
    true_position_list = []
    action_cov = np.diag([0.25, 0.25, 0.1])
    meas_cov = np.diag([0.01,0.01, 0.001])
    true_pos = np.zeros((3,1))
    
    for i in range(5000):
        # Randomly chosen control action
        freq = 1/5000. * np.pi
        r = 1.0
        u = np.array( [ [r], [0.], [i * freq] ] ) #(np.random.rand(2,1) * 2)# - 1
        true_pos = motion_model_odometry_non_linear(u,true_pos)
        true_position_list.append(true_pos[:2].flatten())

        # Generate noise
        u_noise = np.random.multivariate_normal(mean = u.flatten(), cov=action_cov).reshape(3,1)
        z_noise = np.random.multivariate_normal(mean = true_pos.flatten(), cov=meas_cov).reshape(3,1)

        # Estimation
        kf.predict(u_noise,action_cov)
        kf.update(z_noise, meas_cov)
        est_pos = kf.get_mean()
        est_position_list.append(est_pos[:2].flatten())
    est_position_array = np.vstack(est_position_list)
    true_position_array = np.vstack(true_position_list)
    mean_rmsd = np.average( np.linalg.norm( true_position_array - est_position_array, axis = 1 ) )
    print(mean_rmsd)
    plt.plot(est_position_array[:,0], est_position_array[:,1], label="Estimation")
    plt.plot(true_position_array[:,0], true_position_array[:,1], label="True")
    plt.legend()
    plt.show()
    exit(0)