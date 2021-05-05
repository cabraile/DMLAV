from typing import Any
import numpy as np
from modules.pdf import multivariate_gaussian_pdf
from modules.features.feature import Feature

class GlobalPositioningFeature(Feature):

    def __init__(self, position : np.array, covariance = np.array ):
        """
        Parameters
        ===============
        position: numpy.array.
            The array of the position where the feature is referenced.
        covariance: numpy.array.
            The array of the measurement covariance.
        """
        self.position = position.flatten()
        self.cov = covariance.reshape(self.position.shape[0],self.position.shape[0])
        return
    
    def measurement_model( self, position : np.array ) -> float:
        """ 
        Provides the likelihood of measuring the feature given the coordinates of the state.

        Parameters
        =======
        position : numpy.array.
            The state coordinates
        
        Returns
        ==========
        float.
            The likelihood of measuring the feature given the particle's position.
        """
        l = multivariate_gaussian_pdf(self.position, position, self.cov)
        return l