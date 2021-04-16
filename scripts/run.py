from modules.demo.map_loader_yaml                            import load_ways_from_dict, load_landmarks
from modules.demo.data_manager                               import DataManager
from modules.demo.visualization                              import Visualization
from modules.perception.feature_extractors.cnn_feature_extractor    import CNNFeatureExtractor
from modules.perception.sign_detection.detector                     import Detector
from modules.perception                                             import data_association
from modules.filters.mcl_filter.mcl_dml_filter          import MCLDMLFilter
from modules.metrics                                    import FilterMetrics

import numpy as np
import pandas as pd
import yaml
import utm
import os

# TODO list:
# * Set a random seed for reproducibility purposes
# * Further modularization. Maybe create a superclass and implement from it?
# * Check external access of the Filter class. Maybe wrap for the dataset run?
# * df_append_non_duplicates: put it on another file?
# * Load args from yamls but instead of storing in single class variables, store whole dict
# * Nomenclature: actually these are not the filters. Both MCLDML and GABDML are localization methods.

class Filter: 

    def __init__(self):

        self.cfg_dir = os.path.dirname(os.path.realpath(__file__)) + "/config"
        filter_cfg_path = self.cfg_dir + "/filter.yaml"
        perception_cfg_path = self.cfg_dir + "/perception.yaml"
        visualization_cfg_path = self.cfg_dir + "/visualization.yaml"
        dataset_cfg_path = self.cfg_dir + "/dataset.yaml"
        self.args_from_yamls(filter_cfg_path,perception_cfg_path, dataset_cfg_path, visualization_cfg_path)

        # MCL
        self.mcl_dml_filter = MCLDMLFilter()

        # Metrics
        self.metrics = FilterMetrics()

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

    def args_from_yamls(self, filter_cfg_path : str, perception_cfg_path : str, dataset_cfg_path : str, visualization_cfg_path : str):
        with open(filter_cfg_path, "r") as f_yaml:
            args = yaml.load(f_yaml, Loader=yaml.FullLoader)
            self.n_particles = args["n_particles"]
            self.odom_variance = args["odom_variance"]
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
            self.landmarks = Filter.df_append_non_duplicates(self.landmarks, route_landmarks)
            self.landmarks.reset_index(inplace=True,drop=True)
        self.viz.draw_route(idx, self.mcl_dml_filter.routes[idx])
        self.viz.update_landmarks( np.vstack(self.landmarks["coordinates"]) )
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
        self.groundtruth = np.array([x,y])
        return

    def odom_callback(self, data: dict):
        """ Performs prediction on the particles given the odometry data. """
        odom_x = data["odometer"]["x"]
        odom = np.sum(odom_x)
        variance = len(data["odometer"]) * self.odom_variance
        self.mcl_dml_filter.predict(odom, variance)
        return

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

    def image_callback(self, data : dict):
        """ Performs weight update on the particles given data and resamples."""
        # Find best correspondence landmark
        image = data["image"]
        self.landmark_association(image)
        self.segment_feature_association(image)
        return

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

    def compute_metrics(self, curr_ts):
        pc = self.mcl_dml_filter.get_particles_as_pointcloud()
        is_localized = self.mcl_dml_filter.check_is_localized()

        # Update the time metrics
        self.metrics.set_current_ts(curr_ts, is_localized)
        
        # Update the root mean squared error metrics
        if is_localized:
            avg_position = np.average(pc,axis=0)
            if(self.groundtruth is not None):
                rmsd = np.linalg.norm( avg_position - self.groundtruth )
                self.metrics.append_error(rmsd)
        return

    def update_interface(self):
        # Metrics
        elapsed_time = self.metrics.get_ellapsed_time()
        time_localized = self.metrics.get_time_localized()
        time_localized_prop = self.metrics.get_time_proportion_localized() * 100
        mean_rmsd = self.metrics.get_rmsd_mean()
        std_rmsd = self.metrics.get_rmsd_std()
        max_rmsd = self.metrics.get_rmsd_max()
        min_rmsd = self.metrics.get_rmsd_min()
        title_txt = f"Ellapsed time: {elapsed_time:.2f}s. RMSD: {mean_rmsd:.2f}m(Std: {std_rmsd:.2f}; Max:  {max_rmsd:.2f}; Min: {min_rmsd:.2f}). Time localized: {time_localized:.2f}s ({time_localized_prop:.1f}%)."
        
        # Estimations
        pointcloud = self.mcl_dml_filter.get_particles_as_pointcloud()

        # Graphics
        self.viz.update_title( title_txt )
        self.viz.update_landmarks(np.vstack(self.landmarks["coordinates"]))
        self.viz.update_particles(pointcloud)
        if(self.groundtruth is not None):
            self.viz.update_groundtruth(self.groundtruth)
        self.viz.flush()
        return

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

        # Compute metrics
        self.compute_metrics(curr_ts)

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

    branch_xy_dict = {}
    for i in branch_latlon_dict.keys():
        lat, lon = branch_latlon_dict[i]
        x,y, _, _ = utm.from_latlon(lat, lon)
        branch_xy_dict[i] = np.array([x,y])

    branch_ids_to_expand = [1,2,3]

    # Filter init
    mcl_filter = Filter()
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
        
    exit(0)