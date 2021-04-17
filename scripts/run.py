from modules.demo.map_loader_yaml                            import load_ways_from_dict, load_landmarks
from modules.demo.data_manager                               import DataManager
from modules.demo.visualization                              import Visualization
from modules.perception.feature_extractors.cnn_feature_extractor    import CNNFeatureExtractor
from modules.perception.sign_detection.detector                     import Detector
from modules.perception                                             import data_association
from modules.filters.mcl_filter.mcl_dml_filter          import MCLDMLFilter
from modules.filters.ekf.ekf_pose_2d                    import EKFPose2D
from modules.metrics                                    import FilterMetrics

import numpy as np
import pandas as pd
import yaml
import utm
import os

# TODO list:
# * Further modularization. Maybe create a superclass and implement from it?
# * Check external access of the Wrapper class. Maybe wrap for the dataset run?
# * df_append_non_duplicates: put it on another file?
# * Load args from yamls but instead of storing in single class variables, store whole dict
# * Nomenclature: actually these are not the filters. Both MCLDML and GABDML are localization methods.

class Wrapper: 

    def __init__(self):

        np.random.seed(0)

        self.cfg_dir = os.path.dirname(os.path.realpath(__file__)) + "/config"
        filter_cfg_path = self.cfg_dir + "/filter.yaml"
        perception_cfg_path = self.cfg_dir + "/perception.yaml"
        visualization_cfg_path = self.cfg_dir + "/visualization.yaml"
        dataset_cfg_path = self.cfg_dir + "/dataset.yaml"
        self.args_from_yamls(filter_cfg_path,perception_cfg_path, dataset_cfg_path, visualization_cfg_path)

        # MCL
        self.mcl_dml_filter = MCLDMLFilter()
        self.trajectories = {}
        # EKF
        # "mcl_odom" : odom + mcl
        # "gps_only" : gps
        # "mcl_gps" : mcl + gps
        # "mcl_gps_odom" : mcl_gps_odom
        self.ekf_types = ["odom_only","mcl_odom", "gps_only", "mcl_gps", "mcl_gps_odom"]
        self.ekf_types_odom = ["odom_only", "mcl_odom", "mcl_gps_odom"]
        self.ekf_types_gps = ["gps_only", "mcl_gps", "mcl_gps_odom"]
        self.ekf_types_mcl = ["mcl_odom", "mcl_gps", "mcl_gps_odom"]
        self.visualize_type = "mcl_gps_odom"
        self.ekf = {}
        for ekf_type in self.ekf_types:
            self.ekf[ekf_type] = None # Will be initialized with the first position of the groundtruth
            self.trajectories[ekf_type] = []
        # Metrics

        self.metrics = {} 
        for key in self.ekf_types:
            self.metrics[key] = FilterMetrics()

        # Data
        self.data_manager  = DataManager(self.dataset_dir)
        self.groundtruth = None
        self.groundtruth_trajectory = []

        # Data association
        self.fex = CNNFeatureExtractor((self.extractor_im_size,self.extractor_im_size))
        self.sign_detection = Detector(threshold=self.sign_detection_threshold, flag_use_cpu=self.sign_detection_use_cpu, reset_cache=False)

        # Map
        self.ways = {}
        self.landmarks = pd.DataFrame( columns=[ "uid", "name", "timestamp", "coordinates", "path", "rgb", "features"])
        
        # Interface Init
        self.viz = Visualization()
        return

    def init_ekf(self, lat : float, lon : float, yaw : float, cov_diagonal : np.array):
        x,y,_,_ = utm.from_latlon(lat, lon)
        init_state = np.array([x, y, yaw]).reshape(3,1)
        init_cov = np.diag([self.init_variance, self.init_variance, self.init_variance])
        for key in self.ekf.keys():
            self.ekf[key] = EKFPose2D(init_state, init_cov, motion_model_type="odometry", measurement_model_type="position")
        return

    # INPUT
    # ============================

    def args_from_yamls(self, filter_cfg_path : str, perception_cfg_path : str, dataset_cfg_path : str, visualization_cfg_path : str):
        with open(filter_cfg_path, "r") as f_yaml:
            args = yaml.load(f_yaml, Loader=yaml.FullLoader)
            self.n_particles = args["n_particles"]
            self.odom_variance = args["odom_variance"]
            self.fuse_gps = args["fuse_gps"]
        with open(perception_cfg_path, "r") as f_yaml:
            args = yaml.load(f_yaml, Loader=yaml.FullLoader)
            self.extractor_im_size = args["landmark_matcher_image_size"]
            self.landmark_match_range_radius = args["landmark_matcher_range_radius"]
            self.landmark_match_accuracy = args["landmark_matcher_accuracy"]
            self.sign_detection_threshold = args["sign_detection_threshold"]
            self.sign_detection_sensitivity = args["sign_detection_sensitivity"]
            self.sign_detection_fpr = args["sign_detection_fpr"]
            self.sign_detection_use_cpu = args["sign_detection_use_cpu"]
        with open(visualization_cfg_path, "r") as f_yaml:
            args = yaml.load(f_yaml, Loader=yaml.FullLoader)
            self.skip_every_n_frames = args["skip_every_n_frames"]
        with open(dataset_cfg_path, "r") as f_yaml:
            args = yaml.load(f_yaml, Loader=yaml.FullLoader)
            self.dataset_dir = args["dataset_dir"]
            self.routes_dir = args["routes_dir"]
            self.init_route_id = args["init_route_id"]
            self.init_position = args["init_position"]
            self.init_variance = args["init_variance"]
            self.start_timestamp = args["start_timestamp"]
        return

    def load_route_info(self, path : str, idx : int):
        """
        Retrieve the route information from YAML files

        Parameters
        ========
        path: str.
            The absolute path to the route's YAML file.
        idx: int.
            The index of the route.
        """
        route_info = None
        with open(path, "r") as f:
            route_info = yaml.load(f, Loader=yaml.FullLoader)
        idx = int(idx)
        route = load_ways_from_dict(route_info["ways"], flag_to_utm=True)
        self.mcl_dml_filter.add_route(idx, route)
        route_landmarks = load_landmarks(route_info["landmarks"], self.fex, uids=self.landmarks["uid"].to_numpy())
        if(route_landmarks is not None):
            self.landmarks = Wrapper.df_append_non_duplicates(self.landmarks, route_landmarks)
            self.landmarks.reset_index(inplace=True,drop=True)
        self.viz.draw_route(idx, self.mcl_dml_filter.routes[idx])
        self.viz.update_landmarks( np.vstack(self.landmarks["coordinates"]) )
        return

    # ============================

    # DATA ASSOCIATION
    # ======================

    def landmark_association(self, image: np.array) -> bool:
        features = self.fex.extract(image)
        ds_features = np.vstack(self.landmarks["features"].to_numpy())
        m_id, m_s = data_association.match(features, ds_features)
        m_img = self.landmarks["rgb"].to_numpy()[m_id]
        if(m_s < 0.261):
            return False
        l_xy = (self.landmarks["coordinates"]).to_numpy()[m_id]

        # Update weights and Resample
        self.mcl_dml_filter.update_from_landmark(l_xy, self.landmark_match_range_radius)
        return True

    def segment_feature_association(self, image:np.array) -> bool:
        bboxes, labels, img_viz = self.sign_detection.detect_and_recognize(image)

        # Check if detected anything
        if labels is None:
            return False

        # If any label is different from each other, ignore: conflicting information.
        label_set = set(labels)
        if len(label_set) > 1:
            return False

        # Check if it is a speed limit detection
        det = labels[0]
        if det == "not_park":
            return False
        
        speed_limit = int(det)

        # Update weights and resample
        self.mcl_dml_filter.update_from_segment_feature(speed_limit,self.sign_detection_sensitivity, self.sign_detection_fpr)     
        return True

    # ======================

    # MESSAGE CALLBACKS
    # ======================

    def gps_callback(self, data : dict):
        cov = np.diag([1e0, 1e0])
        lats = data["gps"]["latitude"]
        lons = data["gps"]["longitude"]
        gps_array = np.hstack((lats, lons)).reshape(-1,2)
        for row in range(gps_array.shape[0]):
            lat = gps_array[row,0]
            lon = gps_array[row,1]
            x,y,_,_ = utm.from_latlon(lat,lon)
            z = np.array([x,y]).reshape(2,1)
            # Update only the MCL
            if self.fuse_gps:
                self.mcl_dml_filter.update_from_gps(z.flatten(), cov)
            # Update the EKFs
            for key in self.ekf.keys():
                if not self.fuse_gps:
                    if key in self.ekf_types_gps:
                        self.ekf[key].update(z, cov)
                else:
                    if key == "mcl_only":
                        self.ekf[key].update(z, cov)
        return

    def groundtruth_callback(self, data : dict):
        """
        Stores the groundtruth from the incoming data.

        Parameters
        ========
        data: dict.
            The dictionary in which the groundtruth.
        """
        lat = np.average(data["groundtruth"]["latitude"])
        lon = np.average(data["groundtruth"]["longitude"])
        x, y, _, _ = utm.from_latlon(lat, lon)
        xy = np.array([x,y])
        self.groundtruth = xy
        return

    def odom_callback(self, data: dict):
        """ Performs prediction on the particles given the odometry data. """
        variance = self.odom_variance
        cov = np.diag([1e-2, 1e-2, 1e-3])
        odom_df = data["odometer"]
        xs = odom_df["x"]
        ys = odom_df["y"]
        yaws = odom_df["yaw"]
        odom_array = np.hstack((xs,ys,yaws)).reshape(-1,3)
        for row in range(odom_array.shape[0]):
            odom_x = odom_array[row,0]
            odom_y = odom_array[row,1]
            odom_yaw = odom_array[row,2]
            self.mcl_dml_filter.predict(odom_x, variance)
            u = np.array([odom_x, odom_y, odom_yaw]).reshape(3,1)
            for key in self.ekf.keys():
                if key in self.ekf_types_odom:
                    self.ekf[key].predict(u, cov)
                else:
                    self.ekf[key].predict(np.zeros((3,1)), cov) # Increase uncertainty
        return

    def image_callback(self, data : dict):
        """ Performs weight update on the particles given data and resamples."""
        # Find best correspondence landmark
        image = data["image"]
        updated = self.landmark_association(image)
        updated = self.segment_feature_association(image) or updated
        return

    # ======================

    # OTHERS
    # ============================

    def update_ekf_using_mcl(self):
        if self.mcl_dml_filter.check_is_localized() :
            mean, cov = self.mcl_dml_filter.get_mean_and_covariance()
            if np.abs(np.linalg.det(cov)) < 1e-9:
                vmax= np.abs(np.max(cov))
                cov = np.diag([vmax,vmax])
            for method in self.ekf_types_mcl:
                self.ekf[method].update(mean.reshape(2,1), cov)
        return

    def store_current_poses_to_trajectories(self):
        for method in self.ekf.keys():
            mean = self.ekf[method].get_mean()
            cov = self.ekf[method].get_covariance()
            pose_cov = np.hstack((mean.flatten(), cov.flatten()))
            self.trajectories[method].append(pose_cov)
        self.groundtruth_trajectory.append(self.groundtruth)
        return

    @staticmethod
    def df_append_non_duplicates(a, b):
        """
        Solution from:
        https://stackoverflow.com/questions/21317384/pandas-python-how-to-concatenate-two-dataframes-without-duplicates
        """
        if ((a is not None and type(a) is not pd.core.frame.DataFrame) or (b is not None and type(b) is not pd.core.frame.DataFrame)):
            raise ValueError('a and b must be of type pandas.core.frame.DataFrame.')
        if (a is None):
            return(b)
        if (b is None):
            return(a)
        aind = a["uid"].values
        bind = b["uid"].values
        take_rows = list(set(bind)-set(aind))
        take_rows = [i in take_rows for i in bind]

        return(a.append( b.iloc[take_rows,:] ))

    def skip_timestamps(self, to_timestamp : int):
        """ Skip unwanted timestamps until to_timestamp (in nanoseconds). """
        while(self.data_manager.has_next()):
            data = self.data_manager.next()

            # Stores the last groundtruth received
            if(data["groundtruth"] is not None):
                self.groundtruth_callback(data)

            # Ignore timestamps
            if(data["timestamp"] < to_timestamp):
                print(f"\r> Ignoring data from timestamp {data['timestamp']}", end="")
                continue
            else:
                break
        print("")
        return

    def compute_metrics(self, curr_ts, method):
        is_localized = self.mcl_dml_filter.check_is_localized()

        # Update the time metrics
        self.metrics[method].set_current_ts(curr_ts, is_localized)
        
        avg_pose = self.ekf[method].get_mean().flatten()
        avg_position = avg_pose[:2]
        if(self.groundtruth is not None):
            rmsd = np.linalg.norm( avg_position - self.groundtruth )
            self.metrics[method].append_error(rmsd)
        # Update the root mean squared error metrics
        #if is_localized:
        #    avg_position = np.average(pc,axis=0)
        #    if(self.groundtruth is not None):
        #        rmsd = np.linalg.norm( avg_position - self.groundtruth )
        #        self.metrics[method].append_error(rmsd)
        return

    def update_interface(self):
        # Metrics
        elapsed_time = self.metrics[self.visualize_type].get_ellapsed_time()
        time_localized = self.metrics[self.visualize_type].get_time_localized()
        time_localized_prop = self.metrics[self.visualize_type].get_time_proportion_localized() * 100
        mean_rmsd = self.metrics[self.visualize_type].get_rmsd_mean()
        std_rmsd = self.metrics[self.visualize_type].get_rmsd_std()
        max_rmsd = self.metrics[self.visualize_type].get_rmsd_max()
        min_rmsd = self.metrics[self.visualize_type].get_rmsd_min()
        title_txt = f"Ellapsed time: {elapsed_time:.2f}s. RMSD: {mean_rmsd:.2f}m(Std: {std_rmsd:.2f}; Max:  {max_rmsd:.2f}; Min: {min_rmsd:.2f}). Time localized: {time_localized:.2f}s ({time_localized_prop:.1f}%)."
        
        # Estimations
        pointcloud = self.mcl_dml_filter.get_particles_as_pointcloud()
        e_mean = self.ekf[self.visualize_type].get_mean().flatten()[:2]
        e_cov = self.ekf[self.visualize_type].get_covariance()[:2,:2].reshape(2,2)

        # Graphics
        self.viz.update_title( title_txt )
        self.viz.update_landmarks(np.vstack(self.landmarks["coordinates"]))
        self.viz.update_estimated_position(e_mean,e_cov)
        self.viz.update_particles(pointcloud)
        if(self.groundtruth is not None):
            self.viz.update_groundtruth(self.groundtruth)
        self.viz.flush()
        return

    def flush_results(self, directory : str):
        trajectory_array = np.vstack(self.groundtruth_trajectory)
        np.savetxt(directory + f"/trajectory_groundtruth.csv", trajectory_array, delimiter=",", fmt="%.3f")
        for method in self.ekf.keys():
            trajectory_array = np.vstack(self.trajectories[method])
            np.savetxt(directory + f"/trajectory_ekf_{method}.csv", trajectory_array, delimiter=",", fmt="%.3f")

        with open(directory + "/results.csv", "w") as f_out:
            header = "Method,Elapsed Time (s),Time Localized (s),Time Localized (%),Mean RMSD,Std RMSD,Max RMSD,Min RMSD\n"
            print("-----------------------------------------------------------------------------------------")
            print(header)
            f_out.write(header)
            for method in self.ekf.keys():
                elapsed_time        = self.metrics[method].get_ellapsed_time()
                time_localized      = self.metrics[method].get_time_localized()
                time_localized_prop = self.metrics[method].get_time_proportion_localized() * 100
                mean_rmsd           = self.metrics[method].get_rmsd_mean()
                std_rmsd            = self.metrics[method].get_rmsd_std()
                max_rmsd            = self.metrics[method].get_rmsd_max()
                min_rmsd            = self.metrics[method].get_rmsd_min()
                line = f"{method},{elapsed_time},{time_localized:.2f},{time_localized_prop:.2f},{mean_rmsd:.2f},{std_rmsd:.2f},{max_rmsd:.2f},{min_rmsd:.2f}\n"
                f_out.write(line)
                print(line)
            print("-----------------------------------------------------------------------------------------")
            
        return 

    # ============================

    # CYCLE
    # ============================

    def spin_once(self):
        # Data input
        if not self.data_manager.has_next():
            return False
        data = self.data_manager.next()

        # For computing the time metrics purposes
        curr_ts = data["timestamp"]
        
        # For computing the error metrics
        if data["groundtruth"] is not None:
            self.groundtruth_callback(data)

        # Data used for filtering
        if data["odometer"] is not None:
            self.odom_callback(data)

        if data["image"] is not None:
            self.image_callback(data)
            self.viz.update_camera( self.fex.adjust_image(data["image"]) )

        if data["gps"] is not None:
            self.gps_callback(data)

        # Update position for each ekf method that uses MCL
        self.update_ekf_using_mcl()

        self.store_current_poses_to_trajectories()

        # Compute metrics
        for method in self.ekf_types:
            self.compute_metrics(curr_ts, method)

        # Interface Update
        if(self.data_manager.time_idx % self.skip_every_n_frames == 0):
            self.update_interface()

        return True

    def spin(self):
        """ Performs the cyclic prediction/update/resample steps. """
        # Main loop
        flag_can_spin = True
        while flag_can_spin:
            flag_can_spin = self.spin_once()
        return 

    # ============================

if __name__=="__main__":
    # Setup
    branch_names = {
        1: "branch_alexandrina",
        2: "branch_sao_sebastiao",
        3: "branch_serafim_vieira"
    }

    branch_latlon_dict = {
        1: np.array([-22.01233, -47.89182]),
        2: np.array([-22.01322, -47.88981]),
        3: np.array([-22.00964, -47.90088])
    }

    init_pose_dict = {
        "lat": -22.0113353932100217491552,
        "lon": -47.90081526337215223065868,
        "yaw" : np.pi/2,
        "cov_diagonal" : [1e1,1e1,0.1]
    }

    branch_xy_dict = {}
    for i in branch_latlon_dict.keys():
        lat, lon = branch_latlon_dict[i]
        x,y, _, _ = utm.from_latlon(lat, lon)
        branch_xy_dict[i] = np.array([x,y])

    branch_ids_to_expand = [1,2,3]
    
    # Wrapper init
    mcl_filter = Wrapper()
    mcl_filter.init_ekf(**init_pose_dict)
    mcl_filter.load_route_info(f"{mcl_filter.routes_dir}/init_route.yaml", 0)
    mcl_filter.mcl_dml_filter.sample_on_route(mcl_filter.init_position, mcl_filter.init_variance, mcl_filter.init_route_id, mcl_filter.n_particles)
    mcl_filter.skip_timestamps(mcl_filter.start_timestamp)
    
    steps = 0
    while(mcl_filter.spin_once()):

        steps += 1
        if(steps % 50 != 0):
            continue

        # Check if branch is close to any of the particles
        pointcloud = mcl_filter.mcl_dml_filter.get_particles_as_pointcloud()
        expanded = []
        for exp_idx in range(len(branch_ids_to_expand)):

            branch_id = branch_ids_to_expand[exp_idx]
            xy = branch_xy_dict[branch_id].reshape(1,2)
            diff = pointcloud - xy
            distances = np.linalg.norm(diff, axis=1)
            closest_particle_id = np.argmin(distances)

            # If a particle is close enough to the branch, get its route idx
            # and copy the particles to the other route
            if float(distances[closest_particle_id]) < 10:
                b_name = branch_names[branch_id]
                closest_route = int(mcl_filter.mcl_dml_filter.particles[closest_particle_id, 1])
                mcl_filter.load_route_info(f"{mcl_filter.routes_dir}/{b_name}.yaml", branch_id)
                mcl_filter.mcl_dml_filter.copy_to_route(closest_route, branch_id)
                expanded.append(exp_idx)

        # Remove expanded branches from list
        for exp_idx in expanded:
            branch_ids_to_expand.pop(exp_idx)
        
    mcl_filter.flush_results("C:/Users/carlo/Local_Workspace")
    exit(0)