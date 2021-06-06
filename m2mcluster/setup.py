from __future__ import print_function

#import matplotlib
#matplotlib.use('Agg')

import numpy
from amuse.lab import *
from matplotlib import pyplot
from amuse.units import nbody_system,units
from amuse.community.hermite0.interface import Hermite

def setup_star_cluster(N=100, Mcluster=100.0 | units.MSun, Rcluster= 1.0 | units.parsec, softening=0.1 | units.parsec, W0=0.,imf='kroupa'):

	#Setup nbody converter
	converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
	epsilon2=softening**2.

	if W0==0.:
	    stars=new_plummer_sphere(N,converter)
	else:
	    stars=new_king_model(N,W0,convert_nbody=converter)

	if imf=='kroupa':
	    stars.mass=new_broken_power_law_mass_distribution(N,
	                                               mass_boundaries= [0.08, 0.5, 100] |units.MSun,
	                                               alphas= [-1.3,-2.3] )

	    
	stars.scale_to_standard(convert_nbody=converter, smoothing_length_squared = epsilon2)
	stars.move_to_center()

	return stars
