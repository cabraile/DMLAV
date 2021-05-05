import numpy as np
from modules.filters.ekf.ekf import EKF

class EKFPose2D(EKF):
    """
    Implements the 2D EKF used for fusing odometry measurements and 
    global position (DML methods) and pose estimations (GNSS methods).
    """

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
        """
        The motion model when the control command received is an odometry.

        Parameters 
        ==========
        u: numpy.array.
            The odometry received of shape (3,1)
        x: numpy.array.
            The state before motion of shape (3,1)

        Returns
        ==========
        g_x: numpy.array.
            The state after prediction. Shape (3,1).
        """
        c = np.cos(x[2,0])
        s = np.sin(x[2,0])
        g_x = np.array([
            [x[0,0] + c * u[0,0] - s * u[1,0] ],
            [x[1,0] + s * u[0,0] + c * u[1,0]],
            [x[2,0] + u[2,0]]
        ])
        return g_x
    
    def G_odometry(self, u : np.array, x : np.array) -> np.array:
        """
        The jacobian of the motion model when the control command received is an odometry.

        Parameters 
        ==========
        u: numpy.array.
            The odometry received of shape (3,1)
        x: numpy.array.
            The state before motion of shape (3,1)

        Returns
        ==========
        G_x: numpy.array.
            The jacobian of the motion model. Shape (3,3).
        """
        c = np.cos(x[2,0])
        s = np.sin(x[2,0])
        G_x = np.array([
            [1., 0., -s * u[0,0] - c * u[1,0] ],
            [0., 1., c * u[0,0] - s * u[1,0]],
            [0., 0., 1.]
        ])
        return G_x
    
    def h_position(self, x : np.array) -> np.array:
        """
        The measurement model when the measurement to be estimated is the position.

        Parameters 
        ==========
        x: numpy.array.
            The state for estimating the measurement. Shape =(3,1)

        Returns
        ==========
        The state's (x,y) position. Shape = (2,1).
        """
        return x[0:2, 0].reshape(2,1)

    def H_position(self, x : np.array) -> np.array:
        """
        The jacobian of the measurement model when the measurement to be estimated is the position.

        Parameters 
        ==========
        x: numpy.array.
            The state for estimating the measurement. Shape = (3,1)

        Returns
        ==========
        The 2x2 identity matrix.
        """
        return np.eye(2, x.shape[0])
    
    def h_pose(self, x : np.array) -> np.array:
        """
        The measurement model when the measurement to be estimated is the pose.

        Parameters 
        ==========
        x: numpy.array.
            The state for estimating the measurement. Shape =(3,1)

        Returns
        ==========
        The state's (x,y,yaw) position. Shape = (3,1).
        """
        return x

    def H_pose(self, x : np.array) -> np.array:
        """
        The jacobian of the measurement model when the measurement to be estimated is the pose.

        Parameters 
        ==========
        x: numpy.array.
            The state for estimating the measurement. Shape = (3,1)

        Returns
        ==========
        The 3x3 identity matrix.
        """
        return np.eye(x.shape[0])