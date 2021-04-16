import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def match(query_features : np.array, dataset_features : np.array) -> int:
    """
    Finds the best correspondence to the query input on the dataset.

    Parameters
    =======
    query_features: numpy.array.
        The 1D array (n_features,) of features from the query.
    dataset_features: numpy.array.
        The 2D array (size_dataset, n_features) of the features from the dataset.
    
    Returns
    ==========
    match_id: int.
        The index of the dataset feature that maximizes the similarity score.
    match_similarity: float.
        The similarity between the query and the matched features.
    """
    q = query_features.reshape(1,-1)
    D = dataset_features
    s = cosine_similarity(D,q).flatten()
    match_id = np.argmax(s)
    match_similarity = float(s[match_id])
    return match_id, match_similarity