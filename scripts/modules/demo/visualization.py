import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
from typing import Union

class Visualization:
    """
    Visualization of the estimation process.

    Attributes
    ==============
    axes: dict.
        The dicitionary of axes used for visualization.
        axes["main"] : matplotlib.Axes.
            The main axis. Illustrates the routes, particles and groundtruth in a global scale.
        axes["zoom"] : mpl_toolkits.axes_grid1.inset_locator.inset_axes.
            A zoomed version of axes["main"].
        axes["camera"] : mpl_toolkits.axes_grid1.inset_locator.inset_axes.
            The most recent image received by the camera sensor.
    particles_scatter_plots: dict.
        The dictionary returned from the particles' scatter plots.
        Responsible for updating the position of the particles.
        particles_scatter_plots["main"]: matplotlib.collections.PathCollection.
        particles_scatter_plots["zoom"]: matplotlib.collections.PathCollection.
    groundtruth_scatter_plots: dict.
        The dictionary returned from the groundtruth's scatter plots.
        Responsible for updating the position of the groundtruth.
        groundtruth_scatter_plots["main"]: matplotlib.collections.PathCollection.
        groundtruth_scatter_plots["zoom"]: matplotlib.collections.PathCollection.
    camera_plot : matplotlib.image.AxesImage.
        The container of the most recent image.
    """

    def __init__(self):

        # Init dictionaries
        self.axes = {"main" : None, "zoom" : None, "camera" : None}
        self.routes_plots = {"main" : None, "zoom": None}
        self.particles_scatter_plots = {"main" : None, "zoom" : None}
        self.groundtruth_scatter_plots = {"main" : None, "zoom" : None}
        self.landmarks_scatter_plots = {"main" : None, "zoom" : None}
        self.camera_plot = None
        self.already_plotted_routes = []

        # Iteractive figure
        plt.ion()

        # Instantiation of the figure and the main axis
        self.fig, self.axes["main"] = plt.subplots(figsize=(10,10))

        # Instantiation of the zoomed axis (inset_axes)
        self.axes["zoom"] = inset_axes(self.axes["main"], loc='lower left', height="30%", width="30%")
        self.axes["zoom"].axes.xaxis.set_visible(False)
        self.axes["zoom"].axes.yaxis.set_visible(False)
        mark_inset(self.axes["main"], self.axes["zoom"], loc1=1, loc2=2, fc="none", ec="0.5")

        # Instantiation of the camera axis (inset_axes)
        self.axes["camera"] = inset_axes(self.axes["main"], loc='upper center', height="40%", width="40%")
        self.axes["camera"].axes.xaxis.set_visible(False)
        self.axes["camera"].axes.yaxis.set_visible(False)

        # Instantiation of the scatter plots.
        
        self.particles_scatter_plots["main"] = self.axes["main"].scatter([], [], cmap="rainbow", marker="x", label="Particles") # color = "black"
        self.particles_scatter_plots["zoom"] = self.axes["zoom"].scatter([], [], cmap="rainbow", marker="x")  # color = "black"
        self.groundtruth_scatter_plots["main"] = self.axes["main"].scatter([], [], color="red", s=100, marker="*", label="Groundtruth")
        self.groundtruth_scatter_plots["zoom"] = self.axes["zoom"].scatter([], [], color="red", s=100, marker="*")
        self.landmarks_scatter_plots["main"] = self.axes["main"].scatter([], [], color="green", s=200, marker="v", label="Landmark")
        self.landmarks_scatter_plots["zoom"] = self.axes["zoom"].scatter([], [], color="green", s=200, marker="v")
        self.routes_plots["main"] = self.axes["main"].plot([], [])
        self.routes_plots["zoom"] = self.axes["zoom"].plot([], [])

        # Instantiation of the camera plot
        self.camera_plot = self.axes["camera"].imshow( np.full((128,128,3),255) )

        # Figure details
        plt.xlabel("Easting (in meters)")
        plt.ylabel("Northing (in meters)")
        return

    def get_waypoints(self, way_df : pd.DataFrame) -> Union[list,list]:

        # Fetch the init and end coordinates for each way
        column_init = way_df["p_init"].to_numpy()
        column_end = way_df["p_end"].to_numpy()
        
        # Iterate for each way in the route's ways
        n_ways = way_df.shape[0]
        xs = []; ys = []
        for w_id in range(n_ways):
            x_init, y_init = column_init[w_id]
            x_end, y_end = column_end[w_id]
            xs.append(x_init); xs.append(x_end)
            ys.append(y_init); ys.append(y_end)

        return xs, ys

    def draw_routes(self, ways_dict: dict):
        """ Plot each route of the map. 
        
        Parameters
        ============
        ways_dict: dict.
            The dictionary of dataframes representing the ways for each route.
        """
        # Iterate over each of the routes' ways
        route_ids = ways_dict.keys()
        for route_id in reversed(route_ids):
            way_df = ways_dict[route_id]
            xs, ys = self.get_waypoints(way_df)

            # Plot the coordinates
            self.routes_plots["main"] = self.axes["main"].plot(xs, ys, label=label)
            self.routes_plots["zoom"] = self.axes["zoom"].plot(xs, ys)
        return 

    def draw_route(self, route_id : int, way_df: pd.DataFrame):
        """ Plot a route in the map. 
        
        Parameters
        ============
        """
        # Iterate over each of the routes' ways
        xs, ys = self.get_waypoints(way_df)

        # Plot the coordinates
        self.axes["main"].plot(xs, ys, label=f"Route {route_id}")
        self.axes["zoom"].plot(xs, ys)
        return 

    def update_particles(self, pointcloud : np.array, weights : np.array = None):
        """
        Update the particles' positions in the scatter plot.

        Parameters
        =========
        pointcloud: numpy.array.
            The (n_particles, 3) array that represents set of particles.
            The first and second columns represent the x,y coordinates for each particle.
            The third column represents the current weight of each particle.
        """
        for axis_name in ["main", "zoom"]:
            self.particles_scatter_plots[axis_name].set_offsets(pointcloud)
            if(weights is not None):
                self.particles_scatter_plots[axis_name].set_array(weights)
                self.particles_scatter_plots[axis_name].set_clim([weights.min(), weights.max()])
        
        return

    def update_groundtruth(self, groundtruth : np.array, zoom : float = 50):
        """
        Update the groundtruth' position in the scatter plot.

        Parameters
        =========
        groundtruth: numpy.array.
            The (2,) array that represents the groundtruth.
        zoom: float.
            The distance radius in meters from the groundtruth center.
        """

        x, y = groundtruth
        self.axes["zoom"].set_xlim(x - zoom, x + zoom)
        self.axes["zoom"].set_ylim(y - zoom, y + zoom)
        for axis_type in ["main", "zoom"]:
            self.groundtruth_scatter_plots[axis_type].set_offsets(groundtruth.reshape(1,-1))
        return

    def update_camera(self, image : np.array):
        """ 
        Changes the current image displayed in the placeholder. 
        Parameters
        =======
        image: numpy.array.
            The new image.
        """
        self.camera_plot.set_data(image)
        return

    def update_title(self, title : str):
        plt.title(title)
        return

    def update_landmarks(self, landmarks : np.array):
        """
        Parameters
        ==============
        landmarks: numpy.array.
            The 2D array of x,y coordinates for each landmark.
        """
        for axis_name in ["main", "zoom"]:
            self.landmarks_scatter_plots[axis_name].set_offsets(landmarks)
        return

    def flush(self):
        """Apply changes to the figure."""
        self.axes["main"].legend(loc="upper right")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return