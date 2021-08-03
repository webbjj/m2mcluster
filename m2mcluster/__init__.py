from . import cluster
from . import functions
from . import algorithm
from . import plot
from . import setup

#
#Functions
#

made_to_measure=algorithm.made_to_measure

get_dynamical_time_scale=functions.get_dynamical_time_scale
density=functions.density
mean_squared_velocity=functions.mean_squared_velocity

nbinmaker=functions.nbinmaker
binmaker=functions.binmaker

plot_positions=plot.positions_plot
mean_squared_velocity_profile=plot.mean_squared_velocity_profile
density_profile=plot.density_profile

setup_star_cluster=setup.setup_star_cluster

#Classes
starcluster=cluster.starcluster

