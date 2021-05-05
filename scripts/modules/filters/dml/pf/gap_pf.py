import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from typing import Union

from modules.features.feature import Feature
from modules.filters.dml.map_representation import from_map_representation_to_xy
from modules.features.segment_feature import SegmentFeature
from modules.features.landmark_feature import LandmarkFeature
from modules.features.global_positioning_feature import GlobalPositioningFeature

class GAPPF():
    """
    Implements the 'Gaussian Approximation of the Posterior-based Particle Filter' method
    for digital map-based localization.
    """

    def __init__(self, prune_gamma : float, n_particles : int):
        """
        Parameters
        ==========
        prune_gamma : float.
            The threshold of the relative score for prunning hypotheses.
        n_particles : int.
            The number of particles sampled for the estimation for each hypothesis
        """
        self.routes = {}
        self.hypotheses = {}
        self.n_particles = n_particles
        self.gamma = prune_gamma
        return
    
    # STATE
    # ==========================

    def add_hypothesis(self, mean : float, variance : float, route : int):
        """
        Create a new hypothesis.

        Parameters
        ===============
        mean: float.
            The mean position to of the hypothesis's distribution.
        variance: float.
            The variance of the hypothesis's distribution
        route: int.
            To which route this hypothesis belongs.
        """
        h = np.array([mean, variance], dtype="float").reshape(1,2)
        route = int(route)
        self.hypotheses[route] = h
        return

    def prune_hypothesis(self, route_id : int):
        """
        Remove a hypothesis.

        Parameters
        =========
        route_id: int.
            The identifier of the route from which the hypothesis is going to 
            be removed.
        """
        del self.hypotheses[route_id]
        del self.routes[route_id]
        return

    def check_is_localized(self) -> bool:
        """
        Indicates whether the estimation is reliable (localized) or not.

        Returns
        ====
        bool. 
            True if the estimation is localized (one hypothesis on the set of hypotheses). False otherwise.
        """

        if len(self.hypotheses) == 1:
            return True
        return False

    def get_mean(self) -> np.array:
        """ 
        Get the mean in x,y coordinates. 
        Warning: it does not check if the method is localized! 

        Returns
        =========
        mean: numpy.array.
            The mean (x,y) position of the estimation.
        """
        r_id = list( self.hypotheses )[0]
        hypothesis = self.hypotheses[r_id].flatten()
        mean, variance = hypothesis
        ls = np.random.normal(mean, variance ** 0.5, size=(self.n_particles))
        route = self.routes[r_id]
        xy_list = []
        for l in ls:
            xy = from_map_representation_to_xy(l, route)
            xy_list.append(xy)
        xy_array = np.vstack(xy_list)
        mean = np.mean(xy_array, axis=0)
        return mean
    
    def get_mean_and_covariance(self) -> Union[np.array, np.array] :
        """ 
        Get the mean and covariance in x,y coordinates. 
        Warning: it does not check if the method is localized! 

        Returns
        =========
        mean: numpy.array.
            The mean (x,y) position of the estimation.
        covariance: numpy.array.
            The covariance of the estimation.
        """
        r_id = list( self.hypotheses )[0]
        hypothesis = self.hypotheses[r_id].flatten()
        mean, variance = hypothesis
        ls = np.random.normal(mean, variance ** 0.5, size=(self.n_particles))
        route = self.routes[r_id]
        xy_list = []
        for l in ls:
            xy = from_map_representation_to_xy(l, route)
            xy_list.append(xy)
        xy_array = np.vstack(xy_list)
        mean = np.mean(xy_array, axis=0)
        covariance = EmpiricalCovariance().fit(xy_array).covariance_ + np.full((2,2), 1.)
        return mean, covariance

    # ==========================

    # MAP
    # ==========================
    
    def add_route(self, route_id : int, ways : pd.DataFrame):
        """
        Add a route to the set of routes.

        Parameters
        ===========
        route_id: int.
            The identifier of the route.
        ways: pandas.DataFrame.
            The DataFrame of the information of ways contained in the route.
        """
        self.routes[route_id] = ways
        return
    
    # ==========================

    # HYPOTHESES MANAGEMENT
    # ==========================
    
    def copy_to_route(self, from_idx : int, to_idx : int):
        """
        Copies one hypothesis's distribution to another route.

        Parameters
        =======
        from_idx: int.
            The identifier of the route from which the hypothesis's 
            distribution is going to be copied.
        to_idx: int.
            The identifier of the route to which the hypothesis's
            distribution is going to be copied.
        """
        h = self.hypotheses[from_idx].flatten()
        mean, variance = h
        self.add_hypothesis(mean, variance, to_idx)
        return
    
    # ==========================

    # MOTION
    # ==========================

    def predict(self, odom : float, odom_var : float):
        """
        Performs prediction of the distribution for each hypothesis.

        Parameters
        ==============
        odom: float.
            The forward odometry received.
        odom_var: float.
            The variance of the odometry.
        """
        for r_id in self.routes:
            self.hypotheses[r_id][0,0] += odom
            self.hypotheses[r_id][0,1] += odom_var
        return
    
    # ==========================

    # UPDATE-RELATED TASKS
    # ==========================

    def update(self, feature : Feature) :
        """
        Estimates the new distribution for each hypothesis given a measured feature.
        Performs branch pruning after estimation using the gamma defined at the instantiation.

        Parameters
        ============
        feature: Feature-inherited objects.
            The measured feature detected. 
        """
        N = self.n_particles
        scores = {}
        max_score = None
        for route_id in self.routes:
            likelihoods = np.empty((N,)) 
            h = self.hypotheses[route_id].flatten()
            route = self.routes[route_id]
            mean, variance = h
            l_samples = np.random.normal(mean, variance ** 0.5, size=(N))
            # Compute Likelihoods
            if isinstance(feature,LandmarkFeature) or isinstance(feature, GlobalPositioningFeature):
                for i in range(N):
                    l = l_samples[i]
                    p_xy = from_map_representation_to_xy(l, route)
                    likelihood = feature.measurement_model(p_xy)
                    likelihoods[i] = likelihood
            elif isinstance(feature, SegmentFeature):
                for i in range(N):
                    l = l_samples[i]
                    likelihood = feature.measurement_model(l, route)
                    likelihoods[i] = likelihood
            else:
                raise Exception("Error: feature type provided not implemented!")
            
            # Score each hypothesis
            score = np.sum(likelihoods)

            if max_score is None:
                max_score = score

            max_score = score if (score > max_score) else max_score
            scores[route_id] = score

            # If score is as low as zero, increase hypothesis' variance
            # since it might be on a very different place
            if score < 1e-9:
                self.hypotheses[route_id][0,1] += 1e0
                continue
            eta = 1./score

            # Update their mean and variance
            new_mean = eta * np.sum( likelihoods * l_samples )
            diff = l_samples - new_mean
            new_var = 0.0
            for k in range(N):
                new_var += likelihoods[k] * (diff[k] ** 2.0)
            new_var *= eta

            self.hypotheses[route_id] = np.array([new_mean, new_var]).reshape(1,2)
        
        # If all the particles have a very low score, then do not prune hypotheses!
        if max_score <= 0:
            return

        # Normalize scores and detect routes to prune
        prune_ids = []

        for route_id in self.routes:
            norm_score = scores[route_id] / max_score
            if norm_score < self.gamma:
                prune_ids.append(route_id)
                
        # Prune routes
        for route_id in prune_ids:
            self.prune_hypothesis(route_id)
        return

    # ==========================