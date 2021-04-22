from modules.demo.map_loader_yaml                            import load_ways_from_dict, load_landmarks
from modules.demo.data_manager                               import DataManager
from modules.demo.visualization                              import Visualization
from modules.perception.feature_extractors.cnn_feature_extractor    import CNNFeatureExtractor
from modules.perception.sign_detection.detector                     import Detector
from modules.perception                                             import data_association
from modules.filters.dml.mcl.mcl                        import DMLMCL
from modules.filters.dml.pf.gap_pf                      import GAPPF
from modules.filters.ekf.ekf_pose_2d                    import EKFPose2D
from modules.metrics                                    import FilterMetrics
from modules.features.global_positioning_feature        import GlobalPositioningFeature
from modules.features.landmark_feature                  import LandmarkFeature
from modules.features.segment_feature                   import SegmentFeature
from modules.demo.pipeline_controller                   import PipelineController

import numpy as np
import pandas as pd
import yaml
import utm
import os

# TODO list:
# * Check external access of the Wrapper class. Maybe wrap for the dataset run?
# * df_append_non_duplicates: put it on another file?
# * Load args from yamls but instead of storing in single class variables, store whole dict

DEFAULT_RANDOM_SEED = 0
DEFAULT_PRUNE_GAMMA = 0.3

class Wrapper: 

    def __init__(self):

        np.random.seed(DEFAULT_RANDOM_SEED)

        self.cfg_dir            = os.path.dirname(os.path.realpath(__file__)) + "/config"
        filter_cfg_path         = self.cfg_dir + "/filter.yaml"
        perception_cfg_path     = self.cfg_dir + "/perception.yaml"
        visualization_cfg_path  = self.cfg_dir + "/visualization.yaml"
        dataset_cfg_path        = self.cfg_dir + "/dataset.yaml"
        self.args_from_yamls(filter_cfg_path,perception_cfg_path, dataset_cfg_path, visualization_cfg_path)

        # Filters
        self.dml_mcl = DMLMCL()
        self.dml_gappf = GAPPF(DEFAULT_PRUNE_GAMMA, self.n_particles)
        self.pipeline_controller = PipelineController()

        # Data
        self.data_manager  = DataManager(self.dataset_dir)
        self.groundtruth = None

        # Data association
        self.fex = CNNFeatureExtractor((self.extractor_im_size,self.extractor_im_size))
        self.sign_detection = Detector(threshold=self.sign_detection_threshold, flag_use_cpu=self.sign_detection_use_cpu, reset_cache=False)

        # Map
        self.ways = {}
        self.landmarks = pd.DataFrame( columns=[ "uid", "name", "timestamp", "coordinates", "path", "rgb", "features"])
        
        # Interface Init
        self.viz = Visualization()
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
            self.map_dir = args["map_dir"]
            self.routes_dir = self.map_dir + "/routes"
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
        self.dml_mcl.add_route(idx, route)
        self.dml_gappf.add_route(idx, route)
        route_landmarks = load_landmarks(self.map_dir, route_info["landmarks"], self.fex, uids=self.landmarks["uid"].to_numpy())
        if(route_landmarks is not None):
            self.landmarks = Wrapper.df_append_non_duplicates(self.landmarks, route_landmarks)
            self.landmarks.reset_index(inplace=True,drop=True)
        self.viz.draw_route(idx, self.dml_mcl.routes[idx],use_color_demo=True)
        self.viz.update_landmarks( np.vstack(self.landmarks["coordinates"]) )
        return

    # ============================

    # UPDATES
    # ======================

    def gps_update(self, z, cov):
        if self.fuse_gps:
            feature = GlobalPositioningFeature(z, cov)
            self.dml_mcl.update(feature)
            self.dml_gappf.update(feature)
        self.pipeline_controller.gps_update(z, cov)
        return

    def landmark_update(self, l_xy : np.array):
        std = self.landmark_match_range_radius / 3.0
        cov = np.diag( [ std ** 2, std ** 2 ] )
        landmark = LandmarkFeature(l_xy, cov)
        # Update weights and Resample
        self.dml_mcl.update(landmark)
        self.dml_gappf.update(landmark)
        return

    def speed_limit_update(self, speed_limit : int):
        feature = SegmentFeature(speed_limit, "speed_limit",self.sign_detection_sensitivity, self.sign_detection_fpr)
        # Update weights and resample
        self.dml_mcl.update(feature)
        self.dml_gappf.update(feature)
        return

    # ======================

    # DATA ASSOCIATION
    # ======================

    def detect_landmark(self, image: np.array) -> bool:
        features = self.fex.extract(image)
        ds_features = np.vstack(self.landmarks["features"].to_numpy())
        m_id, m_s = data_association.match(features, ds_features)
        m_img = self.landmarks["rgb"].to_numpy()[m_id]
        if(m_s < 0.261):
            return False
        l_xy = (self.landmarks["coordinates"]).to_numpy()[m_id]
        self.landmark_update(l_xy)
        return True

    def detect_speed_limit_signs(self, image:np.array) -> bool:
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
        self.speed_limit_update(speed_limit)
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
            self.gps_update(z,cov)
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
        self.pipeline_controller.set_current_groundtruth(xy)
        return

    def odom_callback(self, data: dict):
        """ Performs prediction on the particles given the odometry data. """
        variance = self.odom_variance
        cov = np.diag([variance, variance, variance])
        # Load from dataframe
        odom_df = data["odometer"]
        xs = odom_df["x"]
        ys = odom_df["y"]
        yaws = odom_df["yaw"]
        odom_array = np.hstack((xs,ys,yaws)).reshape(-1,3)
        for row in range(odom_array.shape[0]):
            odom_x = odom_array[row,0]
            odom_y = odom_array[row,1]
            odom_yaw = odom_array[row,2]
            u = np.array([odom_x, odom_y, odom_yaw]).reshape(3,1)
            # DM Filters
            self.dml_mcl.predict(odom_x, variance)
            self.dml_gappf.predict(odom_x, variance)
            # EKFs
            self.pipeline_controller.odometry_prediction(u, cov)
        return

    def image_callback(self, data : dict):
        """ Performs weight update on the particles given data and resamples."""
        # Find best correspondence landmark
        image = data["image"]
        updated = self.detect_landmark(image)
        updated = self.detect_speed_limit_signs(image) or updated
        return

    # ======================

    # OTHERS
    # ============================

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

    def update_interface(self):
        
        # Estimations
        mcl_pointcloud = self.dml_mcl.get_particles_as_pointcloud()
        gappf_mean, gappf_cov = self.dml_gappf.get_mean_and_covariance()

        # Graphics
        self.viz.update_landmarks(np.vstack(self.landmarks["coordinates"]))
        self.viz.update_particles(mcl_pointcloud)
        self.viz.update_estimated_position(gappf_mean, gappf_cov, "GAPPF")
        if(self.groundtruth is not None):
            self.viz.update_groundtruth(self.groundtruth)
        self.viz.flush()
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
        localized = self.dml_mcl.check_is_localized()
        mean, cov = self.dml_mcl.get_mean_and_covariance()
        self.pipeline_controller.mcl_update(mean, cov, localized)

        # Update position for each ekf method that uses GAPPF
        localized = self.dml_gappf.check_is_localized()
        mean, cov = self.dml_gappf.get_mean_and_covariance()
        self.pipeline_controller.gappf_update(mean,cov, localized)

        # The estimated states on each of the pipelines is appended so that
        # the trajectory can be evaluated later
        self.pipeline_controller.append_current_state_to_trajectory()

        # Compute metrics
        self.pipeline_controller.compute_metrics()

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
    mcl_filter.pipeline_controller.init_ekf(**init_pose_dict)
    mcl_filter.load_route_info(f"{mcl_filter.routes_dir}/init_route.yaml", 0)
    mcl_filter.dml_mcl.sample_on_route(mcl_filter.init_position, mcl_filter.init_variance, mcl_filter.init_route_id, mcl_filter.n_particles)
    mcl_filter.dml_gappf.add_hypothesis(mcl_filter.init_position, mcl_filter.init_variance, mcl_filter.init_route_id)
    mcl_filter.skip_timestamps(mcl_filter.start_timestamp)
    
    steps = 0
    while(mcl_filter.spin_once()):

        steps += 1
        if(steps % 50 != 0):
            continue

        # Check if branch is close to any of the particles
        pointcloud = mcl_filter.dml_mcl.get_particles_as_pointcloud()
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
                closest_route = int(mcl_filter.dml_mcl.particles[closest_particle_id, 1])
                mcl_filter.load_route_info(f"{mcl_filter.routes_dir}/{b_name}.yaml", branch_id)
                mcl_filter.dml_mcl.copy_to_route(closest_route, branch_id)
                mcl_filter.dml_gappf.copy_to_route(closest_route, branch_id)
                expanded.append(exp_idx)

        # Remove expanded branches from list
        for exp_idx in expanded:
            branch_ids_to_expand.pop(exp_idx)
        
    mcl_filter.pipeline_controller.flush_results(directory = ".results")
    exit(0)