import numpy as np
import utm
import pandas as pd
import yaml
import cv2

from modules.perception.feature_extractors.feature_extractor import FeatureExtractor

def load_ways_from_dict(ways_dict : dict, flag_to_utm : bool = False) -> pd.DataFrame :
    """
    Given the yaml from a route's ways, load to a pandas.DataFrame.

    Parameters
    =========
    way_dict: dict.
        A dict of ways, in which each way is indexed by its seq_id.
    flag_to_utm: bool.
        Whether to convert the coordinates to UTM. 
        If ways are already in UTM, keep it False. Otherwise, set it to True.

    Returns
    =========
    df: pandas.DataFrame.
        The dataframe containing all the information required for each way.
        Columns:
        - seq_id (index): the index of each way is when followed in sequence.
        - p_init (array): the coordinates of the init.
        - p_end (array): the coordinates of the end.
        - cumulative_length (float): the amount of offset up to that way.
        - uid (int) : the unique identifier for the way.
        - speed_limit (int): the speed limit for that way.
    """
    df = pd.DataFrame(columns=["seq_id", "p_init", "p_end", "cumulative_length", "uid", "street_id", "speed_limit"])

    # Load each way in sequence
    cumulative_length = 0
    n_ways = len(ways_dict)
    for seq_id in range(n_ways):
        entry   = ways_dict[seq_id]
        uid     = int(entry["uid"])
        p_init  = np.array( entry["p_init"] )
        p_end   = np.array( entry["p_end"] )
        speed_limit = np.array(entry["maxspeed"])
        street_id    = entry["street_id"]
        
        # Convert coordinates if not converted yet
        if(flag_to_utm):
            easting_init, northing_init, _, _ = utm.from_latlon(p_init[0], p_init[1])
            easting_end, northing_end, _, _ = utm.from_latlon(p_end[0], p_end[1])
            p_init = np.array((easting_init, northing_init))
            p_end = np.array((easting_end, northing_end))
        

        # Append to dataframe
        entry = {
            "seq_id" : seq_id, "uid" : uid , "street_id" : street_id,
            "p_init" : p_init, "p_end" : p_end, "cumulative_length": cumulative_length, 
            "speed_limit" : speed_limit
        }
        df = df.append(entry,ignore_index=True)

        # Increment cumulative_length
        cumulative_length +=np.linalg.norm(p_end - p_init)
    df.set_index("seq_id", inplace=True)
    return df

def load_landmarks(map_dir : str, path_list : list, extractor : FeatureExtractor = None, uids : np.array = None) -> pd.DataFrame :
    """
    Given the list of path for each landmark YAML, load and store to a pandas.DataFrame

    Parameters
    ===========
    map_dir : str.
        The absolute path to the map's directory
    path_list: list.
        The list of str of the path for each of landmark's yaml file wrt the map directory.
    extractor : FeatureExtractor.
        The feature extraction object.
    uids: array-like.
        The uids of already loaded landmarks.

    Returns
    ============
    df: pandas.Dataframe.
        The dataframe in which the loaded information is stored.
    """
    df = pd.DataFrame(columns=[ "uid", "name", "timestamp", "coordinates", "path", "rgb", "features"])
    images_list = []
    for relative_path in path_list:
        path = map_dir + "/" + relative_path
        with open(path, "r") as f:
            landmark_dict = yaml.load(f, Loader=yaml.FullLoader)
        # Coordinate conversion
        uid = landmark_dict["id"]
        if uids is not None:
            if uid in uids:
                continue
        easting_init, northing_init, _, _ = utm.from_latlon( landmark_dict["lat"] , landmark_dict["lon"] )
        coordinates = np.array((easting_init, northing_init))

        # Load image
        image_relative_path = landmark_dict["path"]
        image_path = map_dir + "/" + landmark_dict["path"]
        img_rgb = cv2.imread(image_path)[:,:,::-1]
        entry = {
            "uid" : uid , 
            "name" : landmark_dict["name"], 
            "timestamp" : landmark_dict["timestamp"], 
            "coordinates" : coordinates, 
            "path" : image_path, 
            "rgb" : img_rgb,
            "features" : None
        }
        df = df.append(entry,ignore_index=True)
        images_list.append(img_rgb)
    if(len(images_list) == 0):
        return None
    if extractor is not None:
        descriptors_array = extractor.extract_batch(images_list)
        for i in range(len(images_list)):
            df.at[i,"features"] = descriptors_array[i,:].flatten()
    return df

if __name__=="__main__":
    route_path = "C:/Users/carlo/Local_Workspace/Map/routes/init_route.yaml"
    with open(route_path, "r") as f:
        route_dict = yaml.load(f, Loader=yaml.FullLoader)
    ways_df = load_ways_from_dict(route_dict["ways"], flag_to_utm=True)
    print(ways_df.iloc[0])
    landmarks_df = load_landmarks(route_dict["landmarks"])
    print(landmarks_df.iloc[0])
    exit(0)