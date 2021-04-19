import numpy as np
import pandas as pd
from typing import Union
from sklearn.covariance import EmpiricalCovariance

from modules.filters.dml.map_representation import from_map_representation_to_xy
from modules.features.segment_feature import SegmentFeature
from modules.features.landmark_feature import LandmarkFeature
from modules.features.global_positioning_feature import GlobalPositioningFeature

# TODO LIST:
# * Remove weights from the state. They are only used during update.

class DMBMCL:
    
    def __init__(self):
        self.routes = {}
        self.particles = None
        self.n_particles = 0
        return
    
    # STATE
    # ==========================

    def check_is_localized(self):
        p_routes = self.particles[:,1]
        unique_routes = set(p_routes)
        if len(unique_routes) > 1:
            return False
        return True

    def get_mean(self) -> np.array:
        pointcloud = self.get_particles_as_pointcloud()
        mean = np.mean( pointcloud, axis=0 )
        return 
    
    def get_covariance(self) -> np.array:
        pointcloud = self.get_particles_as_pointcloud()
        covariance = EmpiricalCovariance().fit(pointcloud).covariance_
        return covariance
    
    def get_mean_and_covariance(self) -> Union[np.array, np.array] :
        pointcloud = self.get_particles_as_pointcloud()
        mean = np.mean( pointcloud, axis=0 )
        covariance = EmpiricalCovariance().fit(pointcloud).covariance_
        return mean, covariance


    # ==========================

    # MAP
    # ==========================
    
    def add_route(self, route_id : int, ways : pd.DataFrame):
        self.routes[route_id] = ways
        return
      
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
            x, r, w = self.particles[row_id,:]
            p_coords = from_map_representation_to_xy(x, self.routes[int(r)])
            coords_array[row_id, :] = p_coords
        return coords_array

    # ==========================

    # PARTICLES' MANAGEMENT
    # ==========================

    def sample_on_route(self, mean : float, std : float, route_id : int, n_particles : int):
        """ 
        Sample particles from a gaussian distribution
        
        Parameters
        ==========
        mean: float.
            Mean of the gaussian distribution.
        std: float.
            Standard deviation of the gaussian distribution.
        route_id: int.
            The route index to be assigned for each particle.
        """
        assert route_id in self.routes, "Error: route not initialized yet!"
        #particles = sample_particles(mu = mean, sigma = std, n_particles = n_particles, route_idx = route_id)
        particles_x = np.random.normal(loc=mean, scale=std, size=(n_particles,1))
        particles_r = np.full((n_particles,1), fill_value=route_id ,dtype="uint32")
        particles_w = np.ones((n_particles,1),dtype=float)
        particles = np.hstack((particles_x, particles_r, particles_w))
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
        ids = DMBMCL.low_variance_sampler(weights)
        self.particles = self.particles[ids,:]
        return

    @staticmethod
    def low_variance_sampler(weights : np.array) -> np.array:
        """
        Samples ids proportional to the weights provided.

        Parameters
        ======
        weights: numpy.array.
            The weights of the particles. Will be normalized in this function.

        Returns
        ======
        ids: numpy.array.
            The new ids for sampling the particles with replacement proportional to their weights.
        """
        sum_w = np.sum(weights)
        if(sum_w == 0):
            return np.arange(0, weights.size)
        w = weights/ sum_w
        n_particles = w.size
        delta = 1./n_particles
        r_init = np.random.rand() * delta
        ids = np.zeros((n_particles),dtype="int")
        
        i = 0
        cumulative_w = w[0]
        for k in range(n_particles):
            # The next cumulative weight has to be greater than this
            r = r_init + k * delta
            while r > cumulative_w:
                # Increment the cumulative weight: still not enough
                i += 1
                cumulative_w += w[i]
            ids[k] = i
            
        return ids

    # ==========================

    # MOTION
    # ==========================

    def predict(self, odom : float, odom_var : float):        
        """
        Sample from the motion model (prediction). Changes happen inplace.

        Parameters
        ==========
        particles: numpy.array.
            The (n_particles,3) array of particles (x, route_idx, weights).
        odom: float.
            The forward odometry provided.
        odom_var: float.
            The variance of the odometry.
        """
        #motion_model(self.particles, odom, odom_var)

        n_particles = self.particles.shape[0]

        # Sample the new particles from the motion model
        noise_array = np.random.normal(loc=odom, scale=odom_var ** 0.5, size=(n_particles))
        old_xs = self.particles[:,0]
        new_xs = old_xs + noise_array
        self.particles[:,0] = new_xs
        return
    
    # ==========================

    # UPDATE-RELATED TASKS
    # ==========================

    def update(self, feature) :
        N = self.n_particles
        likelihoods = np.empty((N,)) 
        if isinstance(feature,LandmarkFeature) or isinstance(feature, GlobalPositioningFeature):
            pointcloud = self.get_particles_as_pointcloud()
            for p_idx in range(N):
                p_xy = pointcloud[p_idx,:]
                likelihood = feature.measurement_model(p_xy)
                likelihoods[p_idx] = likelihood
        elif isinstance(feature, SegmentFeature):
            for p_idx in range(N):
                x, r, w = self.particles[p_idx,]
                likelihood = feature.measurement_model(x, self.routes[r])
                likelihoods[p_idx] = likelihood
        else:
            raise Exception("Error: feature type provided not implemented!")
        self.resample(likelihoods)
        return

    # ==========================