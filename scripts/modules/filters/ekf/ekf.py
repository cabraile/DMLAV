import numpy as np

class EKF:

    def __init__(self, prior_mean : np.array, prior_covariance : np.array):
        self.mean = prior_mean
        self.covariance = prior_covariance
        return

    # GETTERS
    # ==============================================

    def get_mean(self) -> np.array:
        return self.mean

    def get_covariance(self) -> np.array:
        return self.covariance

    # ==============================================

    # SETTERS
    # ==============================================

    def set_motion_model(self, g: callable):
        self.g = g
        return

    def set_jacobian_motion_model(self, G : callable):
        self.G = G
        return
    
    def set_measurement_model(self, h: callable):
        self.h = h
        return

    def set_jacobian_measurement_model(self, H : callable):
        self.H = H
        return

    # ==============================================

    # FILTER
    # ==============================================

    def predict(self, u : np.array, u_cov : np.array):
        g = self.g
        G = self.G(u, self.mean)
        self.mean = g(u, self.mean)
        self.covariance = np.dot(np.dot(G, self.covariance), G.T) + np.dot(np.dot(G, u_cov), G.T)
        return

    def update(self, z : np.array, Q :np.array):
        S = self.covariance
        H = self.H(self.mean)
        h = self.h
        I = np.eye(S.shape[0])
        mu = self.mean
        factor = np.linalg.inv( np.dot( H, np.dot(S, H.T) + Q ) )
        K = np.dot( np.dot(S, H.T), factor )
        self.mean = mu + np.dot(K, (z - h(mu)))
        self.covariance = np.dot( ( I - np.dot(K, H) ), S )
        return K

    
    # ==============================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def motion_model_linear(u,x):
        return u + x
    
    def motion_model_linear_jacobian(u,x):
        return np.eye(x.shape[0])
    
    def measurement_model_linear(x):
        return x
    
    def measurement_model_linear_jacobian(x):
        return np.eye(x.shape[0])

    prior_mean = np.zeros((2,1)).reshape(2,1)
    prior_covariance = np.diag([1e-3,1e-3])
    kf = EKF(prior_mean, prior_covariance)
    kf.set_motion_model(motion_model_linear)
    kf.set_jacobian_motion_model(motion_model_linear_jacobian)
    kf.set_measurement_model(measurement_model_linear)
    kf.set_jacobian_measurement_model(measurement_model_linear_jacobian)

    est_position_list = []
    true_position_list = []
    action_cov = np.diag([0.25, 0.25])
    meas_cov = np.diag([0.01,0.01])
    true_pos = np.zeros((2,1))
    
    for i in range(500):
        # Randomly chosen control action
        freq = 1/10. * np.pi
        r = 1.0
        u = np.array( [ [np.cos(i * freq)], [np.sin(i * freq)]] ) #(np.random.rand(2,1) * 2)# - 1
        true_pos += u
        true_position_list.append(true_pos.flatten())

        # Generate noise
        u_noise = np.random.multivariate_normal(mean = u.flatten(), cov=action_cov).reshape(2,1)
        z_noise = np.random.multivariate_normal(mean = true_pos.flatten(), cov=meas_cov).reshape(2,1)

        # Estimation
        kf.predict(u_noise,action_cov)
        kf.update(z_noise, meas_cov)
        est_pos = kf.get_mean()
        est_position_list.append(est_pos.flatten())
    est_position_array = np.vstack(est_position_list)
    true_position_array = np.vstack(true_position_list)
    mean_rmsd = np.average( np.linalg.norm( true_position_array - est_position_array, axis = 1 ) )
    print(mean_rmsd)
    plt.plot(est_position_array[:,0], est_position_array[:,1], label="Estimation")
    plt.plot(true_position_array[:,0], true_position_array[:,1], label="True")
    plt.legend()
    plt.show()
    exit(0)