from __future__ import print_function

#import matplotlib
#matplotlib.use('Agg')

import numpy
from amuse.lab import *
#from matplotlib import pyplot
from amuse.units import nbody_system,units
from amuse.community.hermite.interface import Hermite
from amuse.community.bhtree.interface import BHTree
from amuse.community.gadget2.interface import Gadget2
from amuse.datamodel import Particles

import logging

from .plot import *
from .algorithm import *
from .functions import *
from .setup import setup_star_cluster

try:
    from galpy.util import coords
except:
    import galpy.util.bovy_coords as coords

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

		self.delta_j_tilde=[]

		self.calc_step=calc_step
		self.step=1.

	def add_observable(self,xlower,x,xupper,y,parameter='density',ndim=3,sigma=None,kernel='identifier',extend_outer=False):

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
		elif extend_outer:
			sigma=np.append(sigma,1.0e-10)

		self.observations[parameter]=[xlower,x,xupper,y,parameter,ndim,sigma,kernel]

		self.delta_j_tilde.append(np.zeros(len(x)))

	def initialize_star_cluster(self,N=100, Mcluster=100.0 | units.MSun, Rcluster= 1.0 | units.parsec, softening=0.1 | units.parsec, W0=0.,imf='kroupa', mmin=0.08 | units.MSun, mmax=1.4 | units.MSun, alpha=-1.3, filename = None):

		if filename is not None:
			m,x,y,z,vx,vy,vz=np.loadtxt(filename,unpack=True)
			self.stars=Particles(len(x))
			self.stars.mass = m | units.MSun
			self.stars.x=x | units.parsec
			self.stars.y=y | units.parsec
			self.stars.z=z | units.parsec
			self.stars.vx=vx | units.kms
			self.stars.vy=vy | units.kms
			self.stars.vz=vz | units.kms

			Mcluster=self.stars.total_mass()
			Rcluster=self.stars.virial_radius()
			self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)

		else:

			self.stars,self.converter=setup_star_cluster(N=N, Mcluster= Mcluster, Rcluster= Rcluster, softening=softening, W0=W0,imf=imf, mmin=mmin, mmax=mmax, alpha=alpha)

		self.softening2=softening**2.

		self.tdyn=get_dynamical_time_scale(Mcluster, Rcluster)

		if self.calc_step:
			self.step=self.tdyn.value_in(units.Myr)

		#Default weights
		self.w0=self.stars.mass.value_in(units.MSun)

		return self.stars,self.converter

	def reinitialize_star_cluster(self,mmin=None, mmax=None, mtot=None, nbin=50, bintype='num'):

		if self.debug:
			print('REINITIALIZE:')
			print('Mcluster = ',self.stars.total_mass().value_in(units.MSun))
			print('Rcluster = ',self.stars.virial_radius().value_in(units.parsec))
			print('N = ',len(self.stars))

		#remove stars with masses below mmin
		if mmin is not None:
			indx=self.stars.mass < mmin
			self.stars.remove_particles(self.stars[indx])
			self.w0=self.w0[np.invert(indx)]

			if self.debug:
				print('Remove %i low mass stars' % np.sum(indx))
				print('Mcluster = ',self.stars.total_mass().value_in(units.MSun),np.sum(self.stars.mass.value_in(units.MSun)))
				print('Rcluster = ',self.stars.virial_radius().value_in(units.parsec))
				print('N = ',len(self.stars))

		#Scale mass of cluster so total mass equals mtot
		if mtot is not None:
			mscale=mtot/self.stars.total_mass()
			self.stars.mass*=mscale

			if self.debug:
				print('Scale cluster by %f' % mscale)
				print('Mcluster = ',self.stars.total_mass().value_in(units.MSun))
				print('Rcluster = ',self.stars.virial_radius().value_in(units.parsec))
				print('N = ',len(self.stars))

		#For stars with masses above mmax, split the mass with a new star with opposite position and velocity in the cluster
		if mmax is not None:
			mindx=self.stars.mass > mmax

			if np.sum(mindx) > 0:

				if self.debug:
					print('Resample %i high-mass with total mass %f' % (np.sum(mindx),np.sum(self.stars.mass[mindx].value_in(units.MSun))))

				mnew,xnew,ynew,znew,vxnew,vynew,vznew=self.resample(mindx,nbin=nbin,bintype=bintype)

				new_stars=Particles(len(mnew))
				new_stars.mass=mnew | units.MSun
				new_stars.x=xnew | units.parsec
				new_stars.y=ynew | units.parsec
				new_stars.z=znew | units.parsec
				new_stars.vx=vxnew | units.kms
				new_stars.vy=vynew | units.kms
				new_stars.vz=vznew | units.kms
				new_w0=mnew

				self.stars.remove_particles(self.stars[mindx])
				self.w0=self.w0[np.invert(mindx)]

				self.stars.add_particles(new_stars)
				self.w0=np.append(self.w0,new_w0)

				if self.debug:
					print('Mcluster = ',self.stars.total_mass().value_in(units.MSun))
					print('Rcluster = ',self.stars.virial_radius().value_in(units.parsec))
					print('N = ',len(self.stars))

		self.stars.move_to_center()
		Mcluster=self.stars.total_mass()
		Rcluster=self.stars.virial_radius()

		if self.debug:
			print('DONE:')
			print('Mcluster = ',Mcluster.value_in(units.MSun))
			print('Rcluster = ',Rcluster.value_in(units.parsec))
			print('N = ',len(self.stars))

		self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
		self.tdyn=get_dynamical_time_scale(Mcluster, Rcluster)

		if self.calc_step:
			self.step=self.tdyn.value_in(units.Myr)

	def resample(self,mindx,nbin=50,bintype='num'):

		w0bar=np.mean(self.w0)
		r=np.sqrt((self.stars.x.value_in(units.parsec))**2.+(self.stars.y.value_in(units.parsec))**2.+(self.stars.z.value_in(units.parsec))**2.)

		mnew=np.array([])
		xnew=np.array([])
		ynew=np.array([])
		znew=np.array([])
		vxnew=np.array([])
		vynew=np.array([])
		vznew=np.array([])

		if bintype=='num':
		    rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
		elif bintype =='fix':
		    rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)

		for i in range(0,len(rmid)):
			rindx=(r>=rlower[i])*(r<rupper[i])

			if np.sum(rindx*mindx)>0:

				mtot=np.sum(self.stars.mass[rindx*mindx].value_in(units.MSun))
				ntot=int(np.ceil(mtot/w0bar))

				if self.debug:
					print('RESAMPLE: ',rmid[i],np.sum(rindx*mindx),ntot,mtot)

				x,y,z=self.stars.x[rindx].value_in(units.parsec),self.stars.y[rindx].value_in(units.parsec),self.stars.z[rindx].value_in(units.parsec)
				vx,vy,vz=self.stars.vx[rindx].value_in(units.kms),self.stars.vy[rindx].value_in(units.kms),self.stars.vz[rindx].value_in(units.kms)

				vR,vT,vz=coords.rect_to_cyl_vec(vx,vy,vz,x,y,z)

				r3d=np.random.uniform(rlower[i],rupper[i],ntot)
				phin=2.0*np.pi*np.random.rand(ntot)
				theta=np.arccos(1.0-2.0*np.random.rand(ntot))
				Rn=r3d*np.cos(theta)
				zn=r3d*np.sin(theta)

				vRn=np.random.normal(0.,np.std(vR),ntot)
				vTn=np.random.normal(0.,np.std(vT),ntot)
				vzn=np.random.normal(0.,np.std(vz),ntot)

				xn,yn,zn=coords.cyl_to_rect(Rn,phin,zn)
				vxn,vyn,vzn=coords.cyl_to_rect_vec(vRn,vTn,vzn,phin)

				mnew=np.append(mnew,np.ones(ntot)*w0bar)
				xnew=np.append(xnew,xn)
				ynew=np.append(ynew,yn)
				znew=np.append(znew,zn)

				vxnew=np.append(vxnew,vxn)
				vynew=np.append(vynew,vyn)
				vznew=np.append(vznew,vzn)

		return mnew,xnew,ynew,znew,vxnew,vynew,vznew



		pass

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

	def evaluate(self,epsilon=10.0**-4.,mu=1.,alpha=1.,mscale=1.,zeta=None,xi=None,method='Seyer', **kwargs):
			
		self.stars,self.criteria, self.delta_j_tilde=made_to_measure(self.stars,self.observations,self.w0,epsilon=epsilon,mu=mu,alpha=alpha,mscale=mscale,zeta=zeta,xi=xi,step=self.step,delta_j_tilde=self.delta_j_tilde,method=method,debug=self.debug,**kwargs)

		self.niteration+=1


		return self.stars,self.criteria, self.delta_j_tilde

	def xy_plot(self,filename=None):
		positions_plot(self.stars,filename=filename)

	def rho_prof(self,filename=None):
	    density_profile(self.stars, self.observations,filename=filename)

	def v_prof(self,filename=None):
		mean_velocity_profile(self.stars, self.observations,filename=filename)

	def v2_prof(self,filename=None):
		mean_squared_velocity_profile(self.stars, self.observations,filename=filename)

	def rhov2_prof(self,filename=None):
		density_weighted_mean_squared_velocity_profile(self.stars, self.observations,filename=filename)

	def writeout(self,outfile=None):

		if outfile==None:
			if self.outfile is None:
				self.outfile=open('m2moutfile.dat','w')

			outfile=self.outfile

		if self.niteration==0:
			outfile.write('%i,' % self.niteration)


			for oparam in self.observations:
				rlower,rmid,rupper,obs,param,ndim,sigma,kernel=self.observations[oparam]

				for r in rmid:
					outfile.write('%f,' % r)

				for o in obs:
					outfile.write('%f,' % o)

			outfile.write('%f,' % 0.0)

			for i in range(0,len(self.observations)):
				if i==(len(self.observations)-1):
					outfile.write('%f' % 0.0)
				else:
					outfile.write('%f,' % 0.0)

			outfile.write('\n')

		outfile.write('%i,' % self.niteration)


		c2=np.array([])

		for oparam in self.observations:
			rlower,rmid,rupper,obs,param,ndim,sigma,kernel=self.observations[oparam]


			if param=='rho' or param=='Sigma':
				mod_rho=density(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel)
				for r in rmid:
					outfile.write('%f,' % r)

				for rho in mod_rho:
					outfile.write('%f,' % rho)

				c2=np.append(c2,chi2(obs,mod_rho))

			elif 'v' in param:

				if 'rhov' in param or 'Sigmav' in param:
					mod_v2=density_weighted_mean_squared_velocity(self.stars,rlower,rmid, rupper, param, ndim, kernel=kernel)
				elif 'v' in param and '2' in param:
					mod_v2=mean_squared_velocity(self.stars,rlower,rmid, rupper, param, ndim, kernel=kernel)
				elif 'v' in param and '2' not in param:
					mod_v2=mean_velocity(self.stars,rlower,rmid, rupper, param, ndim, kernel=kernel)

				for r in rmid:
					outfile.write('%f,' % r)

				for v2 in mod_v2:
					outfile.write('%f,' % v2)

				c2=np.append(c2,chi2(obs,mod_v2))


		if self.niteration==0:
			self.criteria=0.

		outfile.write('%f,' % self.criteria)

		for c in c2:

			if c==c2[-1]:
				outfile.write('%f' % c)
			else:
				outfile.write('%f,' % c)

		outfile.write('\n')


	def snapout(self):
		write_set_to_file(self.stars,'%s.csv' % str(self.niteration).zfill(5))



	




