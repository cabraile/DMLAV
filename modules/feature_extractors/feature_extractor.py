import numpy as np
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):

    @abstractmethod
    def __init__(self, input_shape : tuple):
        pass
    
    @abstractmethod
    def prepare(self, image : np.array ):
        pass

    @abstractmethod
    def extract(self, image : np.array ):
        pass