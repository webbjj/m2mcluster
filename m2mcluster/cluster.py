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
from .setup import setup_star_cluster

class starcluster(object):

	def __init__(self, kernel='identifier', calc_step=False,number_of_iterations=100, outfile=None, number_of_workers=1, debug=False,*kwargs):
		self.number_of_iterations=number_of_iterations
		self.number_of_workers=number_of_workers

		self.debug=debug

		self.niteration=0

		self.criteria=0

		self.observations={}

		if outfile==None:
			self.outfile=open('m2moutfile.dat','w')
		else:
			self.outfile=outfile

		self.delta_j_tilde=None

		self.calc_step=calc_step
		self.step=1.

	def add_observable(self,xlower,x,xupper,y,parameter='density',ndim=3,sigma=None,kernel='identifier',rhov2=False,extend_outer=False):

		#'rho' or 'Sigma' for 3d and 2d density
		#'v2','vlos2','vR2','vT2','vz2' for square velocities


		#Add outer bin that extends to 1e10 and has a value near 0
		if extend_outer:
			xlower=np.append(xlower,xupper[-1])
			xupper=np.append(xupper,1e10)
			x=np.append(x,(xupper[-1]-xlower[-1])/2.)
			y=np.append(y,1.0e-10)


		if sigma is None:
			sigma=np.ones(len(x))
		if extend_outer:
			sigma=np.append(sigma,1.0e-10)

		self.observations[parameter]=[xlower,x,xupper,y,parameter,ndim,sigma,kernel,rhov2]

	def initialize_star_cluster(self,N=100, Mcluster=100.0 | units.MSun, Rcluster= 1.0 | units.parsec, softening=0.1 | units.parsec, W0=0.,imf='kroupa', mmin=0.08 | units.MSun, mmax=1.4 | units.MSun, alpha=-1.3):


		self.stars,self.converter=setup_star_cluster(N=N, Mcluster= Mcluster, Rcluster= Rcluster, softening=softening, W0=W0,imf=imf, mmin=mmin, mmax=mmax, alpha=alpha)

		self.softening2=softening**2.

		self.tdyn=get_dynamical_time_scale(Mcluster, Rcluster)

		if self.calc_step:
			self.step=self.tdyn.value_in(units.Myr)

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

		if self.calc_step:
			self.step=self.tdyn.value_in(units.Myr)

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


		print('TIME UNITS: ',tend.as_quantity_in(units.Myr),len(self.stars),self.stars.total_mass().value_in(units.MSun),self.stars.virial_radius().value_in(units.parsec),self.criteria)


		self.gravity_code.particles.add_particles(self.stars)
		self.gravity_code.commit_particles()

		channel_from_stars_to_cluster=self.stars.new_channel_to(self.gravity_code.particles, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])
		channel_from_cluster_to_stars=self.gravity_code.particles.new_channel_to(self.stars, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])

		self.gravity_code.evolve_model(tend)

		channel_from_cluster_to_stars.copy()

		self.gravity_code.stop()

		self.stars.move_to_center()

		return self.stars

	def evaluate(self,epsilon=10.0**-4.,mu=1.,alpha=1.,method='Seyer', **kwargs):
			
		self.stars,self.criteria, self.delta_j_tilde=made_to_measure(self.stars,self.observations,self.w0,epsilon=epsilon,mu=mu,alpha=alpha,step=self.step,delta_j_tilde=self.delta_j_tilde,method=method,debug=self.debug,**kwargs)

		self.niteration+=1


		return self.stars,self.criteria, self.delta_j_tilde

	def xy_plot(self,filename=None):
		positions_plot(self.stars,filename=filename)

	def rho_prof(self,filename=None):
	    density_profile(self.stars, self.observations,filename=filename)

	def v2_prof(self,filename=None):
		mean_squared_velocity_profile(self.stars, self.observations,filename=filename)

	def writeout(self,outfile=None):

		if outfile==None:
			if self.outfile is None:
				self.outfile=open('m2moutfile.dat','w')

			outfile=self.outfile

		if self.niteration==0:
			outfile.write('%i,' % self.niteration)


			for oparam in self.observations:
				rlower,rmid,rupper,obs,param,ndim,sigma,kernel,rhov2=self.observations[oparam]

				for r in rmid:
					outfile.write('%f,' % r)

				for o in obs:
					outfile.write('%f,' % o)

			outfile.write('%f\n' % 0.0)

		outfile.write('%i,' % self.niteration)


		for oparam in self.observations:
			rlower,rmid,rupper,obs,param,ndim,sigma,kernel,rhov2=self.observations[oparam]


			if param=='rho' or param=='Sigma':
				mod_rho=density(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel)
				for r in rmid:
					outfile.write('%f,' % r)

				for rho in mod_rho:
					outfile.write('%f,' % rho)

			if param=='v2' or param=='vlos2' or param=='vR2' or param=='vT2' or param=='vz2':

				mod_v2=mean_squared_velocity(self.stars,rlower,rmid, rupper, param, ndim, kernel=kernel, rhov2=rhov2)

				for r in rmid:
					outfile.write('%f,' % r)

				for v2 in mod_v2:
					outfile.write('%f,' % v2)

		if self.niteration==0:
			self.criteria=0.

		outfile.write('%f\n' % self.criteria)

	def snapout(self):
		write_set_to_file(self.stars,'%s.csv' % str(self.niteration).zfill(5))



	




