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

	def __init__(self, kernel=None, number_of_iterations=100, number_of_workers=1,debug=False,*kwargs):
		self.number_of_iterations=number_of_iterations
		self.number_of_workers=number_of_workers

		self.observed_rho=None
		self.observed_sigv=None

		self.debug=debug

	def add_observable(self,xlower,x,xupper,y,parameter='density',ndim=3):
		self.xlowerobs=xlower
		self.xmidobs=x
		self.xupperobs=xupper
		self.yobs=y
		self.parameter=parameter

		self.ndim=ndim

		if parameter=='density':
			self.observed_rho=[self.xlowerobs,self.xmidobs,self.xupperobs,self.yobs,self.parameter,self.ndim]

		if parameter=='velocity':
			self.observed_sigv=[self.xlowerobs,self.xmidobs,self.xupperobs,self.yobs,self.parameter,self.ndim]

	def initialize_star_cluster(self,N=100, Mcluster=100.0 | units.MSun, Rcluster= 1.0 | units.parsec, softening=0.1 | units.parsec, W0=0.,imf='kroupa', mmin=0.08 | units.MSun, mmax=100 | units.MSun, alpha=-1.3):

		#Setup nbody converter
		self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
		self.softening2=softening**2.

		if W0==0.:
		    self.stars=new_plummer_sphere(N,self.converter)
		else:
		    self.stars=new_king_model(N,W0,convert_nbody=self.converter)

		if imf=='kroupa':
			if mmax <= 0.5 | units.MSun:
				stars=new_powerlaw_mass_distribution(number_of_particles=N,mass_min=mmin,mass_max=mmax,alpha=-1.3)

			elif mmin >=0.5 | units.MSun:
				stars.mass=new_powerlaw_mass_distribution(number_of_particles=N,mass_min=mmin,mass_max=mmax,alpha=-2.3)

			else:
				stars.mass=new_broken_power_law_mass_distribution(N,mass_boundaries= [mmin.value_in(units.MSun), 0.5, mmax.value_in(units.MSun)] | units.MSun,alphas= [-1.3,-2.3],mass_max=mmax )
		
		elif imf=='salpeter':
			stars.mass=new_powerlaw_mass_distribution(number_of_particles=N,mass_min=mmin,mass_max=mmax,alpha=alpha)

		    
		self.stars.scale_to_standard(convert_nbody=self.converter, smoothing_length_squared = self.softening2)
		self.stars.move_to_center()

		self.tdyn=get_dynamical_time_scale(Mcluster, Rcluster)

		#Default weights
		self.w0=self.stars.mass.value_in(units.MSun)

		return self.stars,self.converter

	def reinitialize_star_cluster(self,mmin=None, mmax=None, mtot=None):

		#remove stars with masses below mmin
		if mmin is not None:
			indx=self.stars.mass < mmin
			self.stars.remove_particles(self.stars[indx])
			self.w0=self.w0[np.invert(indx)]

		#Scale mass of cluster so total mass equals mtot
		if mtot is not None:
			mscale=mtot/self.stars.total_mass()
			self.stars.mass*=mscale

		#For stars with masses above mmax, split the mass with a new star with opposite position and velocity in the cluster
		if mmax is not None:
			indx=self.stars.mass > mmax
			new_stars=self.stars[indx].copy_to_new_particles()
			new_stars.x*=-1.
			new_stars.y*=-1.
			new_stars.z*=-1.
			new_stars.vx*=-1.
			new_stars.vy*=-1.
			new_stars.vz*=-1.

			new_w0=self.w0[indx]

			self.stars.add_particles(new_stars)
			self.w0=np.append(self.w0,new_w0)


			indx=self.stars.mass > mmax
			self.stars.mass[indx]/=2.

		self.stars.move_to_center()
		Mcluster=self.stars.total_mass()
		Rcluster=self.stars.virial_radius()
		self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
		self.tdyn=get_dynamical_time_scale(Mcluster, Rcluster)

		print(Mcluster.value_in(units.MSun),Rcluster,self.tdyn)

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


		print('TIME UNITS: ',tend.as_quantity_in(units.Myr))


		self.gravity_code.particles.add_particles(self.stars)
		self.gravity_code.commit_particles()

		channel_from_stars_to_cluster=self.stars.new_channel_to(self.gravity_code.particles, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])
		channel_from_cluster_to_stars=self.gravity_code.particles.new_channel_to(self.stars, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])



		self.gravity_code.evolve_model(tend)

		channel_from_cluster_to_stars.copy()

		self.gravity_code.stop()

		self.stars.move_to_center()

		return self.stars

	def evaluate(self,epsilon=10.0**-4.,mu=1.,alpha=1.,delta_j_tilde=None,kernel=None, plot=False, filename=None, **kwargs):
			
		self.stars,self.criteria, self.delta_j_tilde=made_to_measure(self.stars,self.observed_rho,self.observed_sigv,self.w0,epsilon=epsilon,mu=mu,alpha=alpha,delta_j_tilde=delta_j_tilde,kernel=kernel,debug=self.debug,plot=plot,filename=filename,**kwargs)

		return self.stars,self.criteria, self.delta_j_tilde

	def xy_plot(self,filename=None):
		positions_plot(self.stars,filename=filename)

	def rho_prof(self,filename=None):
	    density_profile(self.stars, self.observed_rho,filename=filename)

	def sigv_prof(self,filename=None):
		velocity_dispersion_profile(self.stars, self.observed_sigv,filename=filename)



	




