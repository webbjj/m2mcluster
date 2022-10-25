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
from amuse.couple import bridge

from clustertools import cart_to_sphere,sphere_to_cart


import logging

from .plot import *
from .algorithm import *

from .functions import *
from .setup import setup_star_cluster

try:
    from galpy.util import coords
except:
    import galpy.util.bovy_coords as coords

from galpy.potential import to_amuse

class starcluster(object):

	def __init__(self, kernel='identifier', calc_step=False,number_of_iterations=100, outfile=None, number_of_workers=1, debug=False,*kwargs):
		self.number_of_iterations=number_of_iterations
		self.number_of_workers=number_of_workers

		self.gravity_code=None

		self.debug=debug

		self.niteration=0

		self.criteria=0

		self.observations={}
		self.models={}
		self.norms={}

		self.outfile=outfile

		self.delta_j_tilde=[]

		self.calc_step=calc_step
		self.step=1.
		self.stars=[]


	def add_observable(self,xlower,x,xupper,y,parameter='density',ndim=3,sigma=None,kernel='identifier',extend_outer=False):

		#'rho' or 'Sigma' for 3d and 2d density
		#'v2','vlos2','vr2','vp2','vt2' for square velocities


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

		if len(self.stars) > 0:
			if parameter=='rho' or parameter=='Sigma':
				mod=density(self.stars,xlower,x,xupper,parameter,ndim,kernel=kernel,**kwargs)
				norm=None

			elif ('rho' in parameter or 'Sigma' in parameter) and ('v' in parameter) and ('2' in parameter):
				mod=density_weighted_mean_squared_velocity(self.stars,xlower,x,xupper,parameter,ndim,kernel=kernel,**kwargs)
				norm=None
			elif ('v' in parameter) and ('2' in parameter):
				mod,norm=mean_squared_velocity(self.stars,xlower,x,xupper,parameter,ndim,kernel=kernel,norm=True,**kwargs)

			self.models[parameter]=mod
			self.norms[parameter]=norm
		else:
			self.models[parameter]=[None]
			self.norms[parameter]=[None]

		self.delta_j_tilde.append(np.zeros(len(x)))

	def initialize_star_cluster(self,N=100, Mcluster=100.0 | units.MSun, Rcluster= 1.0 | units.parsec, softening=0.1 | units.parsec, W0=0.,imf='kroupa', mmin=0.08 | units.MSun, mmax=1.4 | units.MSun, alpha=-1.3, filename = None, **kwargs):

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
			Rcluster=kwargs.get('rv',None)

			if Rcluster is None:
				Rcluster=self.stars.virial_radius()
			self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)

		else:

			self.stars,self.converter=setup_star_cluster(N=N, Mcluster= Mcluster, Rcluster= Rcluster, softening=softening, W0=W0,imf=imf, mmin=mmin, mmax=mmax, alpha=alpha, **kwargs)


		self.ids=np.arange(0,len(self.stars),1)
		self.ntot=len(self.stars)

		self.softening2=softening**2.

		self.tdyn=self.stars.dynamical_timescale()

		if self.calc_step:
			self.step=self.tdyn.value_in(units.Myr)

		#Default weights
		self.w0=self.stars.mass.value_in(units.MSun)

		self.dwdt=np.zeros(len(self.w0))

		if len(self.observations) > 0:
			for j,oparam in enumerate(self.observations):

				rlower,rmid,rupper,obs,param,ndim,sigma,kernel=self.observations[oparam]

				if param=='rho' or param=='Sigma':
					mod=density(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel,**kwargs)
					norm=[None]
				elif ('rho' in param or 'Sigma' in param) and ('v' in param) and ('2' in param):
					mod=density_weighted_mean_squared_velocity(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel,**kwargs)
					norm=[None]
				elif ('v' in param) and ('2' in param):
					mod,norm=mean_squared_velocity(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel,norm=True,**kwargs)

				self.models[oparam]=mod
				self.norms[oparam]=norm

		return self.stars,self.converter

	def restart_star_cluster(self,nsnap,w0,outfilename,softening=0.1 | units.parsec, unit='msunpckms',fmt='standard',**kwargs):

		filename='%s.csv' % str(nsnap).zfill(5)

		self.niteration=nsnap

		try:
			data=np.loadtxt(filename,unpack=True,skiprows=3,delimiter=',')
		except:
			data=np.loadtxt(filename,unpack=True,**kwargs)

		if fmt=='original':
			mass,vx,vy,vz,x,y,z=data.astype(float)
			ids=np.arange(0,len(x),1)
		elif fmt=='dwdt':
			mass,rad,vx,vy,vz,x,y,z,ids,dwdt=data.astype(float)
		elif fmt=='test':
			mass,rad,vx,vy,vz,x,y,z=data.astype(float)
			ids=np.arange(0,len(x),1)
		else:
			mass,vx,vy,vz,x,y,z,ids=data.astype(float)



		if unit=='msunpckms':
			self.stars=Particles(len(x))
			self.stars.mass = mass | units.MSun
			self.stars.x=x | units.parsec
			self.stars.y=y | units.parsec
			self.stars.z=z | units.parsec
			self.stars.vx=vx | units.kms
			self.stars.vy=vy | units.kms
			self.stars.vz=vz | units.kms
		elif unit=='kgmms':
			self.stars=Particles(len(x))
			self.stars.mass = mass | units.kg
			self.stars.x=x | units.m
			self.stars.y=y | units.m
			self.stars.z=z | units.m
			self.stars.vx=vx | units.ms
			self.stars.vy=vy | units.ms
			self.stars.vz=vz | units.ms

		Mcluster=self.stars.total_mass()

		Rcluster=kwargs.get('rv',None)
		if Rcluster is None:
			Rcluster=self.stars.virial_radius()


		self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)

		self.ids=ids
		self.ntot=len(self.stars)

		self.softening2=softening**2.

		self.tdyn=self.stars.dynamical_timescale()

		if self.calc_step:
			self.step=self.tdyn.value_in(units.Myr)

		#Default weights
		if isinstance(w0,float):
			self.w0=np.ones(len(self.stars))*w0
		else:
			self.w0=w0

		self.dwdt=np.zeros(len(self.w0))

		if len(self.observations) > 0:
			for j,oparam in enumerate(self.observations):

				rlower,rmid,rupper,obs,param,ndim,sigma,kernel=self.observations[oparam]

				if param=='rho' or param=='Sigma':
					mod=density(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel,**kwargs)
					norm=None
				elif ('rho' in param or 'Sigma' in param) and ('v' in param) and ('2' in param):
					mod=density_weighted_mean_squared_velocity(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel,**kwargs)
					norm=None
				elif ('v' in param) and ('2' in param):
					mod,norm=mean_squared_velocity(self.stars,rlower,rmid,rupper,param,ndim,kernel=kernel,norm=True,**kwargs)

				self.models[oparam]=mod
				self.norms[oparam]=norm

		outfile=open(outfilename,'r')
		self.outfile=open(outfilename+'.restart','w')

		for i in range(0,nsnap+2):
			line=outfile.readline()
			self.outfile.write(line)

		outfile.close()

		return self.stars,self.converter

	def reinitialize_star_cluster(self,mmin=None, mmax=None, mtot=None, rmax=None, nbin=50, bintype='num',**kwargs):

		if self.debug:
			print('REINITIALIZE:')
			print('N = ',len(self.stars))


		#set masses to stars beyond rmax to zero:
		if rmax is not None:
			r=np.sqrt((self.stars.x.value_in(units.parsec))**2.+(self.stars.y.value_in(units.parsec))**2.+(self.stars.z.value_in(units.parsec))**2.)
			indx=(r>rmax.value_in(units.parsec))

			if np.sum(indx)>0:
				#self.stars.mass[indx]= 0. | units.MSun
				self.stars.remove_particles(self.stars[indx])
				self.w0=self.w0[np.invert(indx)]
				self.ids=self.ids[np.invert(indx)]
				self.dwdt=self.dwdt[np.invert(indx)]

			if self.debug:
				print('Remove %i stars beyond rmax' % np.sum(indx))
				print('N = ',len(self.stars))

		#set masses to stars less than mmin to zero:
		if mmin is not None:
			indx=self.stars.mass < mmin
			#self.stars.mass[rindx]= 0. | units.MSun
			if np.sum(indx)>0:
				self.stars.remove_particles(self.stars[indx])
				self.w0=self.w0[np.invert(indx)]
				self.ids=self.ids[np.invert(indx)]
				self.dwdt=self.dwdt[np.invert(indx)]

			if self.debug:
				print('Remove %i low mass stars' % np.sum(indx))
				print('N = ',len(self.stars))

		#Scale mass of cluster so total mass equals mtot
		if mtot is not None:
			mscale=mtot/self.stars.total_mass()
			self.stars.mass*=mscale

			if self.debug:
				print('Scale cluster by %f' % mscale)
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
				self.ids=self.ids[np.invert(mindx)]
				self.dwdt=self.dwdt[np.invert(mindx)]


				self.stars.add_particles(new_stars)
				self.w0=np.append(self.w0,new_w0)
				self.ids=np.append(self.ids,np.arange(self.ntot+1,self.ntot+1+len(new_stars),1))
				self.ntot+=len(new_stars)
				self.dwdt=np.append(self.dwdt,np.zeros(len(new_stars)))

				if self.debug:
					print('N = ',len(self.stars))

		self.stars.move_to_center()
		Mcluster=self.stars.total_mass()

		Rcluster=kwargs.get('rv',None)
		if Rcluster is None:
			Rcluster=self.stars.virial_radius()

		if self.debug:
			print('DONE:')
			print('Mcluster = ',Mcluster.value_in(units.MSun))
			print('Rcluster = ',Rcluster.value_in(units.parsec))
			print('N = ',len(self.stars))

		self.converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
		self.tdyn=self.stars.dynamical_timescale()

		if self.calc_step:
			self.step=self.tdyn.value_in(units.Myr)

		self.update_models()

	def resample(self,mindx,nbin=50,bintype='num',rscatter=0.,vscatter=0.):

		w0bar=np.mean(self.w0)

		x,y,z=self.stars.x.value_in(units.parsec),self.stars.y.value_in(units.parsec),self.stars.z.value_in(units.parsec)
		vx,vy,vz=self.stars.vx.value_in(units.kms),self.stars.vy.value_in(units.kms),self.stars.vz.value_in(units.kms)

		r,phi,theta,vr,vp,vt=cart_to_sphere(x,y,z,vx,vy,vz)

		mnew=np.array([])
		xnew=np.array([])
		ynew=np.array([])
		znew=np.array([])
		vxnew=np.array([])
		vynew=np.array([])
		vznew=np.array([])

		for i in range(0,np.sum(mindx)):
			mtot=self.stars.mass[mindx][i].value_in(units.MSun)
			ntot=int(np.ceil(mtot/w0bar))
			rn=np.ones(ntot)*r[mindx][i]
			phin=2.0*np.pi*np.random.rand(ntot)
			thetan=np.arccos(1.0-2.0*np.random.rand(ntot))

			vrn=np.ones(ntot)*vr[mindx][i]
			vpn=np.ones(ntot)*vp[mindx][i]
			vtn=np.ones(ntot)*vt[mindx][i]

			if rscatter!=0.:
				rn*=(1.0+(2.0*np.random.rand(ntot)*rscatter-rscatter))
			if vscatter!=0.:
				vrn*=(1.0+(2.0*np.random.rand(ntot)*vscatter-vscatter))
				vpn*=(1.0+(2.0*np.random.rand(ntot)*vscatter-vscatter))
				vtn*=(1.0+(2.0*np.random.rand(ntot)*vscatter-vscatter))

			xn,yn,zn,vxn,vyn,vzn = sphere_to_cart(rn,phin,thetan,vrn,vpn,vtn)

			mnew=np.append(mnew,np.ones(ntot)*w0bar)
			xnew=np.append(xnew,xn)
			ynew=np.append(ynew,yn)
			znew=np.append(znew,zn)

			vxnew=np.append(vxnew,vxn)
			vynew=np.append(vynew,vyn)
			vznew=np.append(vznew,vzn)				


		"""

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

				rad,phi,theta,vr,vp,vt=cart_to_sphere(x,y,z,vx,vy,vz)

				rn=np.random.uniform(rlower[i],rupper[i],ntot)
				phin=2.0*np.pi*np.random.rand(ntot)
				thetan=np.arccos(1.0-2.0*np.random.rand(ntot))
			
				vrn=np.random.normal(np.mean(vr),np.std(vr),ntot)
				vpn=np.random.normal(np.mean(vp),np.std(vp),ntot)
				vtn=np.random.normal(np.mean(vt),np.std(vt),ntot)

				xn,yn,zn,vxn,vyn,vzn = sphere_to_cart(rn,phin,thetan,vrn,vpn,vtn)

				mnew=np.append(mnew,np.ones(ntot)*w0bar)
				xnew=np.append(xnew,xn)
				ynew=np.append(ynew,yn)
				znew=np.append(znew,zn)

				vxnew=np.append(vxnew,vxn)
				vynew=np.append(vynew,vyn)
				vznew=np.append(vznew,vzn)
		"""

		return mnew,xnew,ynew,znew,vxnew,vynew,vznew



		pass

	def update_models(self,return_norm=False):

		models={}
		norms={}

		for j,oparam in enumerate(self.observations):

			rlower,rmid,rupper,obs,param,ndim,sigma,kernel=observations[oparam]

			if param=='rho' or param=='Sigma':
				mod=density(stars,rlower,rmid,rupper,param,ndim,kernel=kernel,**kwargs)
				norm=None
			elif ('rho' in param or 'Sigma' in param) and ('v' in param) and ('2' in param):
				mod=density_weighted_mean_squared_velocity(stars,rlower,rmid,rupper,param,ndim,kernel=kernel,**kwargs)
				norm=None
			elif ('v' in param) and ('2' in param):
				mod,norm=mean_squared_velocity(stars,rlower,rmid,rupper,param,ndim,kernel=kernel,norm=True,**kwargs)

			models[oparam]=mod
			norms[oparam]=norm

		self.models=models
		self.norms=norms

		if return_norm:
			return self.models,self.norms
		else:
			return self.models


	def initialize_gravity_code(self,gravity_code, dt=0.1 | units.Myr, **kwargs):

		self.bridge=False

		if self.gravity_code is not None:
			self.channel_from_stars_to_cluster.copy()
			self.gravity_code.parameters.timestep=dt
			if self.bridge: self.dtbridge=dt/2

		else:

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

			elif gravity_code=='Galpy':
				self.bridge=True

				pot=kwargs.get('pot')
				self.galaxy_code=to_amuse(pot)
				self.gravity_code=drift_without_gravity(convert_nbody=self.converter)
				self.dtbridge=dt/2

			self.gravity_code.particles.add_particles(self.stars)
			if not self.bridge: self.gravity_code.commit_particles()

			self.channel_from_stars_to_cluster=self.stars.new_channel_to(self.gravity_code.particles, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])
			self.channel_from_cluster_to_stars=self.gravity_code.particles.new_channel_to(self.stars, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])

			if self.bridge:
				self.gravity_bridge=bridge.Bridge(use_threading=False)
				self.gravity_bridge.add_system(self.gravity_code, (self.galaxy_code,))
				self.gravity_bridge.timestep = self.dtbridge

	def evolve(self,tend=1. | units.Myr, pot = None):


		print('TIME UNITS: ',tend.value_in(units.Myr),len(self.stars),self.stars.total_mass().value_in(units.MSun),self.criteria)

		if self.bridge:
			self.gravity_bridge.evolve_model(tend)
		else:
			self.gravity_code.evolve_model(tend)

		self.channel_from_cluster_to_stars.copy()

		"""
		if self. bridge:
			self.gravity_bridge.stop()
		else:
			self.gravity_code.stop()
		"""

		self.stars.move_to_center()

		return self.stars

	def evaluate(self,epsilon=10.0**-4.,mu=1.,alpha=1.,**kwargs):

		if kwargs.get("update_models",False):
			self.update_models()
		

		self.stars,self.criteria, self.delta_j_tilde,self.dwdt=made_to_measure(self.stars,self.observations,self.models,self.norms,self.w0,epsilon=epsilon,mu=mu,alpha=alpha,step=self.step,delta_j_tilde=self.delta_j_tilde,debug=self.debug,**kwargs)
		self.niteration+=1


		return self.stars,self.criteria,self.delta_j_tilde,self.dwdt

	def xy_plot(self,filename=None):
		positions_plot(self.stars,filename=filename)

	def rho_prof(self,filename=None):
	    density_profile(self.stars, self.observations,self.models,filename=filename)

	def v_prof(self,filename=None):
		mean_velocity_profile(self.stars, self.observations,self.models,filename=filename)

	def v2_prof(self,filename=None):
		mean_squared_velocity_profile(self.stars, self.observations,self.models,filename=filename)

	def rhov2_prof(self,filename=None):
		density_weighted_mean_squared_velocity_profile(self.stars, self.observations,self.models,filename=filename)

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
			mod=self.models[oparam]

			for r in rmid:
				outfile.write('%f,' % r)

			for m in mod:
				outfile.write('%f,' % m)

			c2=np.append(c2,chi2(obs,m))


		if self.niteration==0:
			self.criteria=0.

		outfile.write('%f,' % self.criteria)

		for i,c in enumerate(c2):

			if i==len(c2)-1:
				outfile.write('%f' % c)
			else:
				outfile.write('%f,' % c)

		outfile.write('\n')


	def snapout(self, return_dwdt=False):

		m=self.stars.mass.value_in(units.MSun)
		x=self.stars.x.value_in(units.parsec)
		y=self.stars.y.value_in(units.parsec)
		z=self.stars.z.value_in(units.parsec)
		vx=self.stars.vx.value_in(units.kms)
		vy=self.stars.vy.value_in(units.kms)
		vz=self.stars.vz.value_in(units.kms)
		ids=self.ids

		if return_dwdt:
			np.savetxt('%s.csv' % str(self.niteration).zfill(5),np.column_stack([m,vx,vy,vz,x,y,z,ids,self.dwdt]))

		else:
			np.savetxt('%s.csv' % str(self.niteration).zfill(5),np.column_stack([m,vx,vy,vz,x,y,z,ids]))

		#write_set_to_file(self.stars,'%s.csv' % str(self.niteration).zfill(5))



class drift_without_gravity(object):
    def __init__(self, convert_nbody, time= 0 |units.Myr):
        self.model_time = time
        self.convert_nbody = convert_nbody
        self.particles = Particles()
    def evolve_model(self, t_end):
        dt = t_end - self.model_time
        self.particles.position += self.particles.velocity*dt
        self.model_time = t_end
    @property
    def potential_energy(self):
        return quantities.zero
    @property 
    def kinetic_energy(self):
        return (0.5*self.particles.mass \
                   *self.particles.velocity.lengths()**2).sum()
    def stop(self):
        pass




