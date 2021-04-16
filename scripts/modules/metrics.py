import numpy as np

class FilterMetrics:

    def __init__(self):
        self.total_rmsd_error = None
        self.rmsd_error_count = 0
        self.rmsd_error_min = None
        self.rmsd_error_max = None
        self.rmsd_error_list = []
        self.time_localized = 0.0
        self.last_localized_ts = None
        self.init_ts = None
        self.curr_ts = None
        return

    def set_current_ts(self, ts : int, is_localized : bool):
        # Set the current timestamp
        if(self.init_ts is None):
            self.init_ts = ts
        self.curr_ts = ts

        # Update the amount of time the algorithm was localized
        if is_localized:
            if(self.last_localized_ts is not None):
                self.time_localized += (self.curr_ts - self.last_localized_ts) * 1e-9
            self.last_localized_ts = self.curr_ts
        else:
            self.last_localized_ts = None
        return
    
    def append_error(self, rmsd_error : float):
        # Update the mean of the root mean squared error
        if(self.total_rmsd_error is None):
            self.rmsd_error_count = 1
            self.total_rmsd_error = rmsd_error
        else:
            self.rmsd_error_count += 1
            self.total_rmsd_error += rmsd_error

        # Update the maximum and minimum metrics
        self.rmsd_error_min = min(rmsd_error, self.rmsd_error_min) if self.rmsd_error_min is not None else rmsd_error
        self.rmsd_error_max = max(rmsd_error, self.rmsd_error_max) if self.rmsd_error_max is not None else rmsd_error
        
        # Append to the error list
        self.rmsd_error_list.append(rmsd_error)
        return

    def get_ellapsed_time(self) -> float:
        assert self.init_ts is not None, "Error: data was still not received!"
        el_time_s = (self.curr_ts - self.init_ts) * 1e-9
        return el_time_s
    
    def get_time_localized(self) -> float:
        return self.time_localized

    def get_time_proportion_localized(self) -> float:
        el_time_s = self.get_ellapsed_time()
        if el_time_s == 0:
            return 1.0
        return self.time_localized / el_time_s

    def get_rmsd_mean(self) ->float:
        assert self.rmsd_error_count != 0, "Error: no errors were appended yet!"
        return self.total_rmsd_error / self.rmsd_error_count

    def get_rmsd_min(self) ->float:
        return self.rmsd_error_min

    def get_rmsd_max(self) ->float:
        return self.rmsd_error_max

    def get_rmsd_std(self) -> float:
        num = np.sum ( ( np.array(self.rmsd_error_list) - self.get_rmsd_mean() ) ** 2.0 ) 
        den = self.rmsd_error_count - 1
        std = np.sqrt(num / den)
        return std
