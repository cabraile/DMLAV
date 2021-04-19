import pandas as pd
from typing import Any

class SegmentFeature():

    def __init__(self, information : Any, info_type : str, model_sensitivity : float, model_fpr : float):
        """
        Parameters
        ===============
        information : Any.
            The object to be compared as a segment feature. Must be comparable.
        info_type: str.
            The type of information this feature carries. It is limited to the
            columns of the dataframe of the map. For instance, "speed_limit".
        model_sensitivity: float.
            The sensitivity of the model which detected the feature.
        model_fpr : float.
            The false positive rate of the model which detected the feature.
        """
        self.info               = information
        self.info_type          = info_type
        self.model_sensitivity  = model_sensitivity
        self.model_fpr          = model_fpr
        self.norm               = 1. / (model_fpr + model_sensitivity)
        self.norm_sensitivity   = self.norm * self.model_sensitivity
        self.norm_fpr           = self.norm * self.model_fpr
        return
    
    def measurement_model( self, accumulated_offset : float, route : pd.DataFrame ) -> float:
        """ 
        Provides the likelihood of measuring the feature given a particle's position and a route.

        Parameters
        =======
        accumulated_offset: float.
            In the (l,r) representation, the l value - which corresponds to the accumulated offset 
            since the estimation process started.
        route: pandas.Dataframe.
            The dataframe containing the information of each way in which the state is represented.
        
        Returns
        ==========
        float.
            The likelihood of measuring the feature given the particle.
        """
        # Find which way the particle belongs
        N_ways = len(route)
        seq_id = 0
        l = accumulated_offset
        for seq_id in range(N_ways-1):
            c_bef = route.at[seq_id, "cumulative_length"] 
            c_aft = route.at[seq_id+1, "cumulative_length"] 
            if l >= c_bef and l < c_aft:
                break

        # Fetches the true speed limit at that way
        true_value = route.at[seq_id, self.info_type]

        # Computes the likelihood
        if true_value != self.info:
            return self.norm_fpr
        return self.norm_sensitivity