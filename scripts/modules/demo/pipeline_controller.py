from modules.filters.ekf.ekf_pose_2d                    import EKFPose2D
from modules.metrics                                    import FilterMetrics
import utm
import numpy as np
import os
class PipelineController:

    def __init__(self):

        self.pipeline_names = [
            "Dead Reckoning","GNSS",
            "MCL",
            "MCL + Odometry",
            "MCL + GNSS (A)", "MCL + GNSS + Odometry (A)",
            "MCL + GNSS (B)", "MCL + GNSS + Odometry (B)",
            "GAPPF",
            "GAPPF + Odometry", 
            "GAPPF + GNSS (A)", "GAPPF + GNSS + Odometry (A)", 
            "GAPPF + GNSS (B)", "GAPPF + GNSS + Odometry (B)"
        ]
        self.pipelines_names_that_use_MCL = [
            "MCL", "MCL + Odometry", 
            "MCL + GNSS (A)", "MCL + GNSS + Odometry (A)",
            "MCL + GNSS (B)", "MCL + GNSS + Odometry (B)"
        ]
        self.pipelines_names_that_use_GAPPF = [
            "GAPPF", 
            "GAPPF + Odometry", 
            "GAPPF + GNSS (A)", "GAPPF + GNSS + Odometry (A)", 
            "GAPPF + GNSS (B)", "GAPPF + GNSS + Odometry (B)"
        ]
        self.pipeline_names_that_use_odometry = [
            "Dead Reckoning",
            "MCL + Odometry", "MCL + GNSS + Odometry (A)", "MCL + GNSS + Odometry (B)",
            "GAPPF + Odometry", "GAPPF + GNSS + Odometry (A)", "GAPPF + GNSS + Odometry (B)"
        ]
        # The approaches (A) fuse GNSS data on the filters, but not in the EKF
        self.pipeline_names_that_use_gnss = [
            "GNSS", 
            "MCL + GNSS (B)", "MCL + GNSS + Odometry (B)",
            "GAPPF + GNSS (B)", "GAPPF + GNSS + Odometry (B)"
        ]

        # Check if all names are consistent
        all_names = set( self.pipeline_names_that_use_odometry + self.pipeline_names_that_use_gnss +\
                    self.pipelines_names_that_use_MCL + self.pipelines_names_that_use_GAPPF )
        differ = all_names ^ set ( self.pipeline_names ) 
        assert  all_names == set ( self.pipeline_names ), f"Error: some names differs from the pipeline names: {differ}"

        # Init the EKF and trajectory dictionaries, but not instantiate any yet
        self.ekf = {}
        self.trajectories = {}
        self.metrics = {}
        for p_name in self.pipeline_names:
            self.ekf[p_name] = None # Will be initialized with the first position of the groundtruth
            self.trajectories[p_name] = []
            self.metrics[p_name] = FilterMetrics()

        # Also store for the groundtruth
        self.groundtruth = None
        self.groundtruth_trajectory = []
        return

     # PREDICTION
    
    def init_ekf(self, lat : float, lon : float, yaw : float, cov_diagonal : np.array):
        x,y,_,_ = utm.from_latlon(lat, lon)
        init_state = np.array([x, y, yaw]).reshape(3,1)
        init_cov = np.diag(cov_diagonal)
        for p_name in self.pipeline_names:
            self.ekf[p_name] = EKFPose2D(init_state, init_cov, motion_model_type="odometry", measurement_model_type="position")
        return

    def append_current_state_to_trajectory(self):
        for p_name in self.pipeline_names:
            mean = self.ekf[p_name].get_mean()
            cov = self.ekf[p_name].get_covariance()
            pose_cov = np.hstack((mean.flatten(), cov.flatten()))
            self.trajectories[p_name].append(pose_cov)
        self.groundtruth_trajectory.append(self.groundtruth)
        return

    def set_current_groundtruth(self, groundtruth : np.array):
        self.groundtruth = groundtruth
        return

    # PREDICTION
    # ======================

    def odometry_prediction(self, u, cov):
        for p_name in self.pipeline_names:
            if p_name in self.pipeline_names_that_use_odometry:
                self.ekf[p_name].predict(u, cov)
            else:
                self.ekf[p_name].predict(np.zeros((3,1)), cov) # Increase uncertainty
        return

    # ======================

    # UPDATES
    # ======================

    def gps_update(self, z, cov):
        """
        Parameters
        ==============
        z: numpy.array.
            The UTM coordinates (x,y).
        cov: numpy.array.
            The covariance of the measurement.
        """
        for p_name in self.pipeline_names_that_use_gnss:
            self.ekf[p_name].update(z,cov)
        return

    def mcl_update(self, mean, cov, is_localized):
        if is_localized:
            # Covariance cannot be close to zero - so underestimate the MCL measurement is a possibility.
            if np.abs(np.linalg.det(cov)) < 1e-9:
                vmax= np.abs(np.max(cov))
                cov = np.diag([vmax,vmax])
            for p_name in self.pipelines_names_that_use_MCL:
                self.ekf[p_name].update(mean.reshape(2,1), cov)
        return

    def gappf_update(self, mean, cov, is_localized):
        if is_localized:
            # Covariance cannot be close to zero - so underestimate the MCL measurement is a possibility.
            if np.abs(np.linalg.det(cov)) < 1e-9:
                vmax= np.abs(np.max(cov))
                cov = np.diag([vmax,vmax])
            for p_name in self.pipelines_names_that_use_GAPPF:
                self.ekf[p_name].update(mean.reshape(2,1), cov)
        return

    # METRICS
    # ======================

    def compute_metrics(self):
        for p_name in self.pipeline_names:
            avg_pose = self.ekf[p_name].get_mean().flatten()
            avg_position = avg_pose[:2]
            if(self.groundtruth is not None):
                rmsd = np.linalg.norm( avg_position - self.groundtruth )
                self.metrics[p_name].append_error(rmsd)
        return
    
    def flush_results(self, directory : str):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        trajectory_array = np.vstack(self.groundtruth_trajectory)
        np.savetxt(directory + f"/trajectory_groundtruth.csv", trajectory_array, delimiter=",", fmt="%.3f")
        for p_name in self.pipeline_names:
            trajectory_array = np.vstack(self.trajectories[p_name])
            np.savetxt(directory + f"/trajectory_pipeline_{p_name}.csv", trajectory_array, delimiter=",", fmt="%.3f")

        with open(directory + "/results.csv", "w") as f_out:
            header = "Method,Mean RMSD,Std RMSD,Max RMSD,Min RMSD\n"
            print("-----------------------------------------------------------------------------------------")
            print(header)
            f_out.write(header)
            for p_name in self.pipeline_names:
                mean_rmsd           = self.metrics[p_name].get_rmsd_mean()
                std_rmsd            = self.metrics[p_name].get_rmsd_std()
                max_rmsd            = self.metrics[p_name].get_rmsd_max()
                min_rmsd            = self.metrics[p_name].get_rmsd_min()
                line = f"{p_name},{mean_rmsd:.2f},{std_rmsd:.2f},{max_rmsd:.2f},{min_rmsd:.2f}\n"
                f_out.write(line)
                print(line)
            print("-----------------------------------------------------------------------------------------")
            
        return 