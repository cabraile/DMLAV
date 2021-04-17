import numpy as np
from modules.filters.ekf.ekf import EKF

class EKFPose2D(EKF):

    def __init__(self, prior_mean : np.array, prior_covariance : np.array, motion_model_type : str, measurement_model_type : str):
        """
        Parameters
        ===================
        motion_model_type: str.
            Which motion model to use: odometry ("odometry") or velocity-based ("velocity").
        measurement_model_type: str.
            Which measurement model to use: position only ("position") or pose ("pose").
        """
        EKF.__init__(self, prior_mean, prior_covariance)
        if motion_model_type == "odometry":
            self.g = self.g_odometry
            self.G = self.G_odometry
        elif motion_model_type == "velocity":
            # TODO
            pass
        else:
            raise Exception(f"Argument 'motion_model_type'={motion_model_type} not implemented.")
        if measurement_model_type == "position":
            self.h = self.h_position
            self.H = self.H_position
        elif measurement_model_type == "pose":
            self.h = self.h_pose
            self.H = self.H_pose
        else:
            raise Exception(f"Argument 'measurement_model_type'={measurement_model_type} not implemented.")
        return

    def g_odometry(self, u : np.array, x : np.array) -> np.array:
        c = np.cos(x[2,0])
        s = np.sin(x[2,0])
        g_x = np.array([
            [x[0,0] + c * u[0,0] - s * u[1,0] ],
            [x[1,0] + s * u[0,0] + c * u[1,0]],
            [x[2,0] + u[2,0]]
        ])
        return g_x
    
    def G_odometry(self, u : np.array, x : np.array) -> np.array:
        c = np.cos(x[2,0])
        s = np.sin(x[2,0])
        G_x = np.array([
            [1., 0., -s * u[0,0] - c * u[1,0] ],
            [0., 1., c * u[0,0] - s * u[1,0]],
            [0., 0., 1.]
        ])
        return G_x
    
    def h_position(self, x : np.array) -> np.array:
        return x[0:2, 0].reshape(2,1)

    def H_position(self, x : np.array) -> np.array:
        return np.eye(2, x.shape[0])
    
    def h_pose(self, x : np.array) -> np.array:
        return x

    def H_pose(self, x : np.array) -> np.array:
        return np.eye(x.shape[0])