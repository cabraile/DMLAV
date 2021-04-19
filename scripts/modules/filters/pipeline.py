
from modules.filters.ekf.ekf_pose_2d                    import EKFPose2D

def Pipeline:

    def __init__(self, prior_mean, prior_covariance):
        self.ekf = EKFPose2D(prior_mean, prior_covariance, motion_model_type="odometry", measurement_model_type="position")
        self.flag_use_odometry = False
        return

    def set_prediction_source(self, source : str):
        if source == "odometry":
            self.flag_use_odometry = True
        # TODO: ADD OTHER SOURCES
        return

    def set_update_source(self, source : str):

        