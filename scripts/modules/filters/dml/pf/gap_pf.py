import numpy as np
from modules.filters.dml.map_representation import from_map_representation_to_xy
from sklearn.covariance import EmpiricalCovariance

class GAPPF():
    
    def __init__(self, prune_gamma):
        self.routes = {}
        self.hypotheses = {}
        self.gamma = prune_gamma
        return
    
    # STATE
    # ==========================

    def add_hypothesis(self, mean : float, variance : float, route : int):
        h = np.array([mean, variance], dtype="float").reshape(1,2)
        route = int(route)
        if(self.hypotheses is None):
            self.hypotheses[route] = h
        else:
            self.hypotheses.append(h)
        return

    def prune_hypothesis(self, route_id):
        del self.hypotheses[route_id]
        del self.routes[route_id]
        return

    def check_is_localized(self):
        if len(self.hypotheses) == 1:
            return True
        return False

    def get_mean(self) -> np.array:
        """ Get the mean in x,y coordinates. Warning: it does not check if the method is localized! """
        r_id, hypothesis = self.hypotheses.values()[0]
        mean, variance = hypothesis
        ls = np.random.normal(mean, variance ** 0.5, size=(20))
        route = self.routes[r_id]
        xy_list = []
        for l in ls:
            xy = from_map_representation_to_xy(l, route)
            xy_list.append(xy)
        xy_array = np.vstack(xy_list)
        mean = np.mean(xy_array, axis=0)
        return mean
    
    def get_mean_and_covariance(self) -> Union[np.array, np.array] :
        """ Get the mean in x,y coordinates. Warning: it does not check if the method is localized! """
        r_id, hypothesis = self.hypotheses.values()[0]
        mean, variance = hypothesis
        ls = np.random.normal(mean, variance ** 0.5, size=(20))
        route = self.routes[r_id]
        xy_list = []
        for l in ls:
            xy = from_map_representation_to_xy(l, route)
            xy_list.append(xy)
        xy_array = np.vstack(xy_list)
        mean = np.mean(xy_array, axis=0)
        covariance = EmpiricalCovariance().fit(pointcloud).covariance_
        return mean, covariance

    # ==========================

    # MAP
    # ==========================
    
    def add_route(self, route_id : int, ways : pd.DataFrame):
        self.routes[route_id] = ways
        return
    
    # ==========================

    # PARTICLES' MANAGEMENT
    # ==========================
    
    def copy_to_route(self, from_idx : int, to_idx : int):
        h = self.hypotheses[from_idx]
        mean, variance = h
        self.add_hypothesis(mean, variance, to_idx)
        return
    
    # ==========================

    # MOTION
    # ==========================

    def predict(self, odom : float, odom_var : float):
        for r_id in self.routes:
            h = self.hypotheses[r_id]
            h[0] += odom
            h[1] += odom_var
        return
    
    # ==========================

    # UPDATE-RELATED TASKS
    # ==========================

    def update(self, feature) :
        N = 50
        scores = {}
        max_score = None
        for route_id in self.routes:
            likelihoods = np.empty((N,)) 
            h = self.hypotheses[route_id]
            route = self.routes[route_id]
            mean, variance = h
            l_samples = np.random.normal(mean, variance ** 0.5, size=(N))
            # Compute Likelihoods
            if isinstance(feature,LandmarkFeature) or isinstance(feature, GlobalPositioningFeature):
                for i in range(N):
                    l = l_samples[i]
                    p_xy = from_map_representation_to_xy(l, route)
                    likelihood = feature.measurement_model(p_xy)
                    likelihoods[p_idx] = likelihood
            elif isinstance(feature, SegmentFeature):
                for i in range(N):
                    l = l_samples[i]
                    likelihood = feature.measurement_model(l, route)
                    likelihoods[p_idx] = likelihood
            else:
                raise Exception("Error: feature type provided not implemented!")
            
            # Score each hypothesis
            score = np.sum(likelihoods)
            if max_score is None:
                max_score = score
            max_score = score if (score > max_score) else max_score
            scores[route_id] = score
            eta = 1./score

            # Update their mean and variance
            new_mean = eta * np.sum( likelihoods * l_samples )
            diff = (l_samples - new_mean).reshape()
            new_var = 0.0
            for k in range(N):
                new_var += likelihoods[k] * (diff[k] ** 2.0)
            new_var *= eta
        
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