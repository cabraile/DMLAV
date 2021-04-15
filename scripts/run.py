from modules.map_loader_yaml                                        import load_ways_from_dict, load_landmarks
from modules.mcl                                                    import sample_particles, motion_model, measurement_model, low_variance_sampler
from modules.data_manager                                           import DataManager
from modules.perception.feature_extractors.cnn_feature_extractor    import CNNFeatureExtractor
from modules.perception.sign_detection.detector         import Detector
from modules.visualization                              import Visualization
from modules                                            import data_association
import numpy as np
import pandas as pd
import yaml
import utm
from sklearn.cluster import KMeans

DEFAULT_MSGS_DIR    = "C:/Users/carlo/Local_Workspace/Datasets/Dataset_DMBL"
DEFAULT_ROUTES_DIR  = "C:/Users/carlo/Local_Workspace/Map/routes"
DEFAULT_INIT_ROUTE  = 0
DEFAULT_INIT_POSITION = 6.0
DEFAULT_INIT_VARIANCE = 10.0
DEFAULT_START_TS    = 1606589686447952032
DEFAULT_VIZ_SKIP_FRAMES = 20

DEFAULT_FEX_IMSIZE = 448
#DEFAULT_FEX_IMSIZE = 224

DEFAULT_N_PARTICLES = 30
DEFAULT_ODOM_VARIANCE = 0.01 # original: 0.09
DEFAULT_DETECTION_RANGE_RADIUS = 5.
DEFAULT_SD_THRESHOLD = 0.85
DEFAULT_SENSITIVITY = 0.9
DEFAULT_FPR = 0.1
DEFAULT_SD_USE_CPU = True

class Filter: 

    def __init__(self, n_particles : int):
        """
        Parameters
        =======
        n_particles: int.
            The number of particles sampled for each route.
        """
        init_route = DEFAULT_INIT_ROUTE

        # MCL
        self.n_particles = n_particles
        self.particles = sample_particles(mu = DEFAULT_INIT_POSITION, sigma = DEFAULT_INIT_VARIANCE ** 0.5, n_particles = self.n_particles, route_idx = init_route)
        self.odom_variance = DEFAULT_ODOM_VARIANCE
        self.extractor_im_size = DEFAULT_FEX_IMSIZE
        self.max_w_particle_id = 0
        self.avg_position = None
        self.mean_abs_error = None
        self.abs_error_sum = None
        self.abs_error_count = 0

        # Data
        self.data_manager  = DataManager(DEFAULT_MSGS_DIR)
        self.groundtruth = None

        # Data association
        self.match_var = DEFAULT_DETECTION_RANGE_RADIUS
        self.match_cov = np.diag([self.match_var, self.match_var])
        self.fex = CNNFeatureExtractor((self.extractor_im_size,self.extractor_im_size))
        self.sign_detection = Detector(threshold=DEFAULT_SD_THRESHOLD, flag_use_cpu=DEFAULT_SD_USE_CPU, reset_cache=False)

        # Map
        self.ways = {}
        self.landmarks = pd.DataFrame( columns=[ "uid", "name", "timestamp", "coordinates", "path", "rgb", "features"])
        
        # Interface Init
        self.viz = Visualization()
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
        self.ways[idx] = load_ways_from_dict(route_info["ways"], flag_to_utm=True)
        route_landmarks = load_landmarks(route_info["landmarks"], self.fex, uids=self.landmarks["uid"].to_numpy())
        if(route_landmarks is not None):
            self.landmarks = Filter.df_append_non_duplicates(self.landmarks, route_landmarks)
            self.landmarks.reset_index(inplace=True,drop=True)
        self.viz.draw_route(idx, self.ways[idx])
        self.viz.update_landmarks( np.vstack(self.landmarks["coordinates"]) )
        return
    
    def copy_to_route(self, from_idx : int, to_idx : int):
        ids_copy = np.where( self.particles[:,1] == from_idx )
        particles_copy = np.copy(self.particles[ids_copy,:]).reshape(-1,3)
        particles_copy[:,1] = to_idx
        stack = np.vstack([self.particles, particles_copy])
        self.particles = stack
        self.n_particles = self.particles.shape[0]
        return

    def particle_to_world_coordinates(self, particle: np.array):
        """
        Convert a particle from (x|r) to the world coordinates.
        
        Parameters
        ===========
        particle: numpy.array.
            The particle's 1D array (x, r, w)
        """
        x, r, w = particle
        r = int(r)
        ways = self.ways[r]
        cumulative_length = ways["cumulative_length"].to_numpy()

        # Checks which in which way the particle belongs
        if x < 0 :
            way_id = 0
        elif x > cumulative_length[-1]:
            way_id = -1
        else:
            for way_id in range(cumulative_length.size - 1):
                if ( cumulative_length[way_id] <= x ) and ( x <= cumulative_length[way_id + 1]  ):
                    break

        # Get the way's information
        row = ways.iloc[way_id]
        p_init = row.at["p_init"]
        p_end = row.at["p_end"]
        p_diff = p_end - p_init

        # Convert to cartesian coordinates
        angle = np.arctan2(p_diff[1], p_diff[0])
        d = x - cumulative_length[way_id]
        delta_array = d * np.array([np.cos(angle),np.sin(angle)])
        p_coords = p_init + delta_array
        return p_coords

    def particles_to_pointcloud(self) -> np.array:
        """
        Provide the particles as a 2D array of their x,y positions.
        Returns
        ==========
        coords_array: numpy.array.
            (n_particles,2) array of the xy positions.
        """
        coords_array = np.empty((self.n_particles,2))
        for row_id in range(self.n_particles):
            particle = self.particles[row_id,:]
            p_coords = self.particle_to_world_coordinates(particle)
            coords_array[row_id, :] = p_coords
        return coords_array

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
        motion_model(self.particles, odom, variance)
        self.max_w_particle_id = np.argmax(self.particles[:,2])
        return

    def landmark_association(self, image: np.array) -> bool:
        features = self.fex.extract(image)
        ds_features = np.vstack(self.landmarks["features"].to_numpy())
        m_id, m_s = data_association.match(features, ds_features)
        m_name = self.landmarks["name"].to_numpy()[m_id] # DEBUG
        m_img = self.landmarks["rgb"].to_numpy()[m_id]
        #if(m_s < 0.392): # 224!
        #if m_s < 0.83: # 448 com GlobalMaxPool
        if(m_s < 0.261): # 448!
            return False
        l_xy = (self.landmarks["coordinates"]).to_numpy()[m_id]

        # Compute likelihood of measurement for each particle
        pointcloud = self.particles_to_pointcloud()
        N = pointcloud.shape[0]
        likelihoods = np.empty((N,)) 
        for p_idx in range(N):
            p_xy = pointcloud[p_idx,:]
            #w = self.particles[p_idx,2]
            #likelihood = w * data_association.measurement_model_landmark(p_xy, l_xy, radius=DEFAULT_DETECTION_RANGE_RADIUS)
            likelihood = data_association.measurement_model_landmark(p_xy, l_xy, radius=DEFAULT_DETECTION_RANGE_RADIUS)
            likelihoods[p_idx] = likelihood

        # Resample
        ids = low_variance_sampler(likelihoods)
        self.particles = self.particles[ids,:]
        self.particles[:,2] = likelihoods[ids]
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

        # Compute likelihood of measurement for each particle
        N = self.particles.shape[0]
        likelihoods = np.empty((N,)) 
        for p_idx in range(N):
            x, r, w = self.particles[p_idx,]
            #likelihood = w * data_association.measurement_model_segment_feature(x, self.ways[r],speed_limit, sensitivity=DEFAULT_SENSITIVITY, fpr=DEFAULT_FPR)
            likelihood = data_association.measurement_model_segment_feature(x, self.ways[r],speed_limit, sensitivity=DEFAULT_SENSITIVITY, fpr=DEFAULT_FPR)
            likelihoods[p_idx] = likelihood

        # Resample
        ids = low_variance_sampler(likelihoods)
        self.particles = self.particles[ids,:]
        self.particles[:,2] = likelihoods[ids]        
        return True

    def check_is_localized(self):
        p_routes = self.particles[:,1]
        unique_routes = set(p_routes)
        if len(unique_routes) > 1:
            return False
        return True

    def image_callback(self, data : dict):
        """ Performs weight update on the particles given data. Performs data association first. """
        # Find best correspondence landmark
        image = data["image"]
        flag_updated = False
        flag_updated = self.landmark_association(image) and flag_updated
        flag_updated = self.segment_feature_association(image) and flag_updated
        if flag_updated:
            self.max_w_particle_id = np.argmax(self.particles[:,2])
        return

    def skip_timestamps(self, to_timestamp):
        """ Skip unwanted timestamps. """
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

    def spin_once(self):
        if not self.data_manager.has_next():
            return False

        data = self.data_manager.next()
        curr_ts = data["timestamp"]

        if data["groundtruth"] is not None:
            self.groundtruth_callback(data)

        if data["odometer"] is not None:
            self.odom_callback(data)

        if data["image"] is not None:
            self.image_callback(data)
            self.viz.update_camera( self.fex.adjust_image(data["image"]) )

        # Compute metrics
        pc = self.particles_to_pointcloud()
        if self.check_is_localized():
            self.avg_position = np.average(pc,axis=0)
            if(self.groundtruth is not None):
                norm = np.linalg.norm( self.avg_position - self.groundtruth )
                if(self.abs_error_sum is None):
                    self.abs_error_count = 1
                    self.abs_error_sum = norm
                else:
                    self.abs_error_count += 1
                    self.abs_error_sum += norm
                self.mean_abs_error = self.abs_error_sum / self.abs_error_count

        # Interface Update
        if(self.data_manager.time_idx % DEFAULT_VIZ_SKIP_FRAMES == 0):
            self.viz.update_title(f"Current timestamp: {curr_ts}s (Error {self.mean_abs_error}m)")
            self.viz.update_landmarks(np.vstack(self.landmarks["coordinates"]))
            ws = self.particles[:,2]
            self.viz.update_particles(pc, ws)
            if(self.groundtruth is not None):
                self.viz.update_groundtruth(self.groundtruth)
            self.viz.flush()

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
    mcl_filter = Filter(n_particles=DEFAULT_N_PARTICLES)
    mcl_filter.load_route_info(f"{DEFAULT_ROUTES_DIR}/init_route.yaml", 0)
    mcl_filter.skip_timestamps(DEFAULT_START_TS)

    steps = 0
    while(mcl_filter.spin_once()):
        steps += 1

        if(steps % 50 != 0):
            continue

        # Check if branch is close to any of the particles
        pointcloud = mcl_filter.particles_to_pointcloud()
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
                closest_route = int(mcl_filter.particles[closest_particle_id, 1])
                mcl_filter.load_route_info(f"{DEFAULT_ROUTES_DIR}/{b_name}.yaml", branch_id)
                mcl_filter.copy_to_route(closest_route, branch_id)
                expanded.append(exp_idx)

        # Remove expanded branches from list
        for exp_idx in expanded:
            branch_ids_to_expand.pop(exp_idx)
        
    exit(0)