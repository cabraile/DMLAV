import numpy as np
import pandas as pd
from modules.filters.mcl_filter.mcl import motion_model, measurement_model, sample_particles, low_variance_sampler, measurement_model_landmark, measurement_model_segment_feature

# TODO LIST:
# * Remove weights from the state. They are only used during update.

class MCLDMLFilter:
    
    def __init__(self):
        self.routes = {}
        self.particles = None
        self.n_particles = 0
        return

    def check_is_localized(self):
        p_routes = self.particles[:,1]
        unique_routes = set(p_routes)
        if len(unique_routes) > 1:
            return False
        return True

    # MAP
    # ==========================
    
    def add_route(self, route_id : int, ways : pd.DataFrame):
        self.routes[route_id] = ways
        return
    
    def from_map_representation_to_xy(self, particle : np.array) -> np.array:
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
        x, r, w = particle
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
    
    def get_particles_as_pointcloud(self) -> np.array:
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
            p_coords = self.from_map_representation_to_xy(particle)
            coords_array[row_id, :] = p_coords
        return coords_array

    # ==========================

    # PARTICLES' MANAGEMENT
    # ==========================

    def sample_on_route(self, mean : float, std : float, route_id : int, n_particles : int):
        assert route_id in self.routes, "Error: route not initialized yet!"
        particles = sample_particles(mu = mean, sigma = std, n_particles = n_particles, route_idx = route_id)
        if( self.particles is None ):
            self.particles = particles
        else:
            self.particles = np.vstack((self.particles, particles))
        self.n_particles = self.particles.shape[0]
        return
    
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