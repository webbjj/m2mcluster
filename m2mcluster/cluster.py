from __future__ import print_function

#import matplotlib
#matplotlib.use('Agg')

import numpy
from amuse.lab import *
from matplotlib import pyplot
from amuse.units import nbody_system,units
from amuse.community.hermite.interface import Hermite
from amuse.community.bhtree.interface import BHTree
from amuse.community.gadget2.interface import Gadget2

import logging

from .plot import *
from .algorithm import *
from .functions import *

class starcluster(object):

	def __init__(self, kernel=None, number_of_iterations=100, number_of_workers=1,*kwargs):
		self.number_of_iterations=number_of_iterations
		self.number_of_workers=number_of_workers
		self.n_iteration=0

	def add_observable(self,xlower,x,xupper,y,parameter='density',ndim=2):
		self.xlowerobs=xlower
		self.xmidobs=x
		self.xupperobs=xupper
		self.yobs=y
		self.parameter=parameter

		self.ndim=ndim

		self.observation=[self.xlowerobs,self.xmidobs,self.xupperobs,self.yobs,self.parameter,self.ndim]


	def initialize_star_cluster(self,N=100, Mcluster=100.0 | units.MSun, Rcluster= 1.0 | units.parsec, softening=0.1 | units.parsec, W0=0.,imf='kroupa'):

		#Setup nbody converter
		self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
		self.softening2=softening**2.

		if W0==0.:
		    self.stars=new_plummer_sphere(N,self.converter)
		else:
		    self.stars=new_king_model(N,W0,convert_nbody=self.converter)

		if imf=='kroupa':
		    self.stars.mass=new_broken_power_law_mass_distribution(N,
		                                               mass_boundaries= [0.08, 0.5, 100] |units.MSun,
		                                               alphas= [-1.3,-2.3] )

		    
		self.stars.scale_to_standard(convert_nbody=self.converter, smoothing_length_squared = self.softening2)
		self.stars.move_to_center()

		self.tdyn=get_dynamical_time_scale(Mcluster, Rcluster)

		return self.stars,self.converter

	def reinitialize_star_cluster(self):
		self.stars.move_to_center()
		Mcluster=self.stars.total_mass()
		Rcluster=self.stars.virial_radius()
		self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
		self.tdyn=get_dynamical_time_scale(Mcluster, Rcluster)

	def initialize_gravity_code(self,gravity_code, dt=0.1 | units.Myr, **kwargs):
		if gravity_code=='BHTree':
			self.gravity_code=BHTree(convert_nbody=self.converter,number_of_workers=self.number_of_workers)
			self.gravity_code.parameters.epsilon_squared = self.softening2
			self.gravity_code.parameters.timestep=dt

			theta=kwargs.get('theta',0.6)
			self.gravity_code.parameters.opening_angle=theta

		elif gravity_code=='Hermite':
			self.gravity_code=Hermite(convert_nbody=self.converter,number_of_workers=self.number_of_workers)
			self.gravity_code.parameters.epsilon_squared = self.softening2
			self.gravity_code.parameters.dt_dia=dt

			dt_param=kwargs.get('dt_param',0.03)
			self.gravity_code.parameters.dt_param=dt_param

		elif gravity_code=='Gadget2':
			self.gravity_code=Gadget2(convert_nbody=self.converter,number_of_workers=self.number_of_workers)
			self.gravity_code.parameters.epsilon_squared = self.softening2

			theta=kwargs.get('theta',0.6)
			self.gravity_code.parameters.opening_angle=theta


	def evolve(self,tend=1. | units.Myr):


		print(self.n_iteration, ' TIME UNITS: ',tend.as_quantity_in(units.Myr))


		self.gravity_code.particles.add_particles(self.stars)
		self.gravity_code.commit_particles()

		channel_from_stars_to_cluster=self.stars.new_channel_to(self.gravity_code.particles, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])
		channel_from_cluster_to_stars=self.gravity_code.particles.new_channel_to(self.stars, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])



		self.gravity_code.evolve_model(tend)

		channel_from_cluster_to_stars.copy()

		self.gravity_code.stop()

		self.stars.move_to_center()

		self.n_iteration+=1

		return self.stars

	def evaluate(self,kernel=None,m2mepsilon=10.0**-4.,debug=False, plot=False, filename=None, **kwargs):
			
		self.stars,self.criteria=made_to_measure(self.stars,self.observation,self.n_iteration,kernel=kernel,m2mepsilon=m2mepsilon,debug=debug,plot=plot,filename=filename,**kwargs)

		return self.stars,self.criteria

	def xy_plot(self,filename=None):
		positions_plot(self.stars,filename=filename)

	def rho_prof(self,filename=None):
	    density_profile(self.stars, self.observation,filename=filename)


	




