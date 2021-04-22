import pandas
import cv2
import os

DEFAULT_CACHE_PATH = "cached_descriptors.csv"

class DataManager:

    def __init__(self, msgs_dir : str):
        """
        Caution: this piece of code assumes each sensor measurement is performed once per timestamp.
        Parameters
        =====
        msgs_dir: . 
            Contains the path of the CSV for each of the sensors.
        """
        self.msgs_dir = msgs_dir
        self.data = {}
        self.cache = None
        ts_dtype = "uint64"
        if (os.path.isfile(DEFAULT_CACHE_PATH)):
            self.cache = pandas.read_csv(DEFAULT_CACHE_PATH, sep=",", dtype={0 : ts_dtype}, header=None,index_col=0)

        # Load odometer measurements to a pandas DataFrame - with timestamp
        self.data["odometer"] = pandas.read_csv(f"{msgs_dir}/odom.csv", sep=",", dtype={"timestamp" : ts_dtype}, index_col="timestamp")

        # Load gps measurements to a pandas DataFrame - with timestamp
        self.data["gps"] = pandas.read_csv(f"{msgs_dir}/gps_raw.csv", sep=",", dtype={"timestamp" : ts_dtype}, index_col="timestamp")

        # Load image path to a pandas Dataframe - with respective timestamps
        self.data["image"] =  pandas.read_csv(f"{msgs_dir}/images.csv", sep=",", dtype={"timestamp" : ts_dtype}, index_col="timestamp")

        # Load groundtruth values to a pandas Dataframe - with respective timestamps
        self.data["groundtruth"] = pandas.read_csv(f"{msgs_dir}/gps_filtered.csv", sep=",", dtype={"timestamp" : ts_dtype}, index_col="timestamp")

        # List all timestamps from received messages during the run
        timestamps = \
            self.data["odometer"].index.tolist() + \
            self.data["image"].index.tolist() + \
            self.data["gps"].index.tolist() + \
            self.data["groundtruth"].index.tolist()

        # Load compass measurements to a pandas DataFrame - with respective timestamps
        self.data["compass"] = {}
        self.data["compass"] = pandas.read_csv(f"{msgs_dir}/orientation.csv", sep=",", index_col="timestamp")
        timestamps += self.data["compass"].index.tolist()

        timestamps = set(timestamps)    # Remove duplicates
        timestamps = list(timestamps)   # Required for sort
        timestamps.sort()               # Timestamps in chronological order
        self.timestamps = timestamps
        self.time_idx = 0
        return

    def next(self) -> dict:
        """
            Retrieves the data for the next time step.
        """
        if(not self.has_next()):
            raise Exception("Error: no new data to be accessed")
        ret = {}
        ts = self.timestamps[self.time_idx]
        ret["timestamp"] = ts
        for data_type in ["odometer", "image", "groundtruth", "compass", "gps"]:
            if(ts in self.data[data_type].index):
                if(data_type != "image"):
                    ret[data_type] = self.data[data_type].loc[ts]
                else:
                    ret[data_type] = None
                    relative_path = self.data[data_type].loc[ts]["path"]
                    path = self.msgs_dir + "/" + relative_path
                    try:
                        ret[data_type] = cv2.imread(path)[:,:,::-1]
                    except:
                        print("\nError: could not load an image from the dataset. Path: '{}'.".format(path))                        
                    if self.cache is not None:
                        if ts in self.cache.index:
                            ret["cache"] = self.cache.loc[ts].values
            else:
                ret[data_type] = None
        self.time_idx+=1
        return ret

    def reset(self):
        self.time_idx = 0
        return

    def has_next(self):
        """
        Checks whether there is still data to be read or if all of the data was already read.

        @return True if there is still data to be read. False otherwise.
        """
        if(self.time_idx < len(self.timestamps)):
            return True
        return False