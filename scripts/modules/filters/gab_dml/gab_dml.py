import numpy as np

class GABDML():
    
    def __init__(self):
        self.routes = {}
        self.hypotheses = []
        return
    
    # STATE
    # ==========================

    def add_hypothesis(self, mean : float, variance : float, route : int):
        
        return

    def check_is_localized(self):
        if len(self.routes) == 1:
            return True
        return False

    def get_mean(self) -> np.array:
        return 
    
    def get_covariance(self) -> np.array:
        return 
    
    def get_mean_and_covariance(self) -> Union[np.array, np.array] :
        return mean, covariance


    # ==========================

    # MAP
    # ==========================
    
    def add_route(self, route_id : int, ways : pd.DataFrame):
        self.routes[route_id] = ways
        return
    
    def from_map_representation_to_xy(self, hypothesis : np.array) -> np.array:
        """
        Convert a particle from (x,r) to the world coordinates.
        
        Parameters
        ===========
        particle: numpy.array.
            The particle's 1D array (x, r, w)
            
        Returns
        ===========
        p_coords : np.array.
            The 1D array of the (x,y) position of the particle.
        """
        x, r, w = hypothesis
        r = int(r)
        ways = self.routes[r]
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
    
    # ==========================

    # PARTICLES' MANAGEMENT
    # ==========================
    
    def copy_to_route(self, from_idx : int, to_idx : int):
        ids_copy = np.where( self.particles[:,1] == from_idx )
        particles_copy = np.copy(self.particles[ids_copy,:]).reshape(-1,3)
        particles_copy[:,1] = to_idx
        stack = np.vstack([self.particles, particles_copy])
        self.particles = stack
        self.n_particles = self.particles.shape[0]
        return
    
    def resample(self, weights : np.array):
        ids = low_variance_sampler(weights)
        self.particles = self.particles[ids,:]
        #self.particles[:,2] = likelihoods[ids]
        return

    # ==========================

    # MOTION
    # ==========================

    def predict(self, odom : float, odom_var : float):        
        motion_model(self.particles, odom, odom_var)
        return
    
    # ==========================

    # UPDATE-RELATED TASKS
    # ==========================

    def update_from_landmark(self, measurement : np.array, detection_range : float):
        pointcloud = self.get_particles_as_pointcloud()
        N = pointcloud.shape[0]
        likelihoods = np.empty((N,)) 
        for p_idx in range(N):
            p_xy = pointcloud[p_idx,:]
            likelihood = measurement_model_landmark(p_xy, measurement, radius = detection_range)
            likelihoods[p_idx] = likelihood

        self.resample(likelihoods)
        return

    def update_from_segment_feature(self, measurement : int, sensitivity : float, fpr : float):
        # Compute likelihood of measurement for each particle
        N = self.particles.shape[0]
        likelihoods = np.empty((N,)) 
        for p_idx in range(N):
            x, r, w = self.particles[p_idx,]
            likelihood = measurement_model_segment_feature(
                x, 
                self.routes[r], 
                measurement, 
                sensitivity = sensitivity, 
                fpr = fpr
            )
            likelihoods[p_idx] = likelihood

        # Resample
        self.resample(likelihoods)   
        return

    # ==========================