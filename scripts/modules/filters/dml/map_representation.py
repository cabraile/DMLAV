import numpy as np
import pandas as pd

def from_map_representation_to_xy(x : float, route : pd.DataFrame) -> np.array:
    """
    Convert a particle from (x,r) to the world coordinates.
    
    Parameters
    ===========
    x: float.
        The accumulated offset of the state.
    route: pandas.DataFrame
        The DataFrame in which the ways are contained. They must be indexed by their sequence id.
        
    Returns
    ===========
    p_coords : np.array.
        The 1D array of the (x,y) position of the particle.
    """
    cumulative_length = route["cumulative_length"].to_numpy()

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
    row = route.iloc[way_id]
    p_init = row.at["p_init"]
    p_end = row.at["p_end"]
    p_diff = p_end - p_init

    # Convert to cartesian coordinates
    angle = np.arctan2(p_diff[1], p_diff[0])
    d = x - cumulative_length[way_id]
    delta_array = d * np.array([np.cos(angle),np.sin(angle)])
    p_coords = p_init + delta_array
    return p_coords