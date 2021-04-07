from modules.map_loader_yaml                            import load_ways_from_dict, load_landmarks
from modules.mcl                                        import sample_particles, motion_model, measurement_model, low_variance_sampler
from modules.data_manager                               import DataManager
from modules.feature_extractors.cnn_feature_extractor   import CNNFeatureExtractor
from modules.visualization                              import Visualization
from modules                                            import data_association
import numpy as np
import pandas as pd
import yaml
import utm

DEFAULT_MSGS_DIR    = "C:/Users/carlo/Local_Workspace/Dataset"
DEFAULT_ROUTES_DIR  = "C:/Users/carlo/Local_Workspace/Map/routes"
DEFAULT_INIT_ROUTE  = 0
DEFAULT_START_TS    = 1606589686447952032
DEFAULT_ODOM_VARIANCE = 0.25
DEFAULT_INIT_VARIANCE = 10.0
DEFAULT_MATCH_VARIANCE = 10.
DEFAULT_N_PARTICLES = 250
DEFAULT_SENSITIVITY = 0.8
DEFAULT_FPR = 0.1

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
        self.particles = sample_particles(mu = 0, sigma = DEFAULT_INIT_VARIANCE ** 0.5, n_particles = self.n_particles, route_idx = init_route)
        self.odom_variance = DEFAULT_ODOM_VARIANCE
        self.extractor_im_size = 224

        # Data
        self.data_manager  = DataManager(DEFAULT_MSGS_DIR)
        self.groundtruth = None

        # Data association
        self.match_var = DEFAULT_MATCH_VARIANCE
        self.match_cov = np.diag([self.match_var, self.match_var])
        self.fex = CNNFeatureExtractor((self.extractor_im_size,self.extractor_im_size))

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
        route_landmarks = load_landmarks(route_info["landmarks"], self.fex)
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
        odom = np.average(data["odometer"]["x"])
        motion_model(self.particles, odom, self.odom_variance)
        return

    def image_callback(self, data : dict):
        """ Performs weight update on the particles given data. Performs data association first. """
        # Find best correspondence landmark
        image = data["image"]
        features = self.fex.extract(image)
        ds_features = np.vstack(self.landmarks["features"].to_numpy())
        m_id, m_s = data_association.match(features, ds_features)
        m_name = self.landmarks["name"].to_numpy()[m_id] # DEBUG
        m_img = self.landmarks["rgb"].to_numpy()[m_id]
        if(m_s < 0.3918):
            return
        l_xy = (self.landmarks["coordinates"]).to_numpy()[m_id]

        # Compute likelihood of measurement for each particle
        pointcloud = self.particles_to_pointcloud()
        N = pointcloud.shape[0]
        likelihoods = np.empty((N,)) 
        for p_idx in range(N):
            p_xy = pointcloud[p_idx,:]
            likelihood = data_association.measurement_model_landmark(p_xy, l_xy, self.match_cov, DEFAULT_SENSITIVITY, DEFAULT_FPR)
            likelihoods[p_idx] = likelihood

        # Resample
        ids = low_variance_sampler(likelihoods)
        self.particles = self.particles[ids,:]
        self.particles[:,2] = likelihoods[ids]
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
            self.viz.update_camera(data["image"])

        # Interface Update
        if(self.data_manager.time_idx % 25 == 0):
            self.viz.update_title(f"Current timestamp: {curr_ts}s")
            self.viz.update_landmarks(np.vstack(self.landmarks["coordinates"]))
            pc = self.particles_to_pointcloud()
            self.viz.update_particles(pc)
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