from __future__ import print_function

import numpy as np
from amuse.lab import *
from amuse.units import nbody_system,units
from amuse.datamodel import Particles

def setup_star_cluster(N=100, Mcluster=100.0 | units.MSun, Rcluster= 1.0 | units.parsec, softening=0.1 | units.parsec, W0=0.,imf='kroupa', mmin=0.08 | units.MSun, mmax=1.4 | units.MSun, alpha=-1.3, filename=None,**kwargs):

	softening2=softening**2.

	if filename is not None:
	    m,x,y,z,vx,vy,vz=np.loadtxt(filename,unpack=True)
	    stars=Particles(len(x))
	    stars.mass = m | units.MSun
	    stars.x=x | units.parsec
	    stars.y=y | units.parsec
	    stars.z=z | units.parsec
	    stars.vx=vx | units.kms
	    stars.vy=vy | units.kms
	    stars.vz=vz | units.kms
	    
	    Mcluster=stars.total_mass()
	    Rcluster=kwargs.get('rv',stars.virial_radius())
	    converter=nbody_system.nbody_to_si(Mcluster,Rcluster)

	else:

		#Setup nbody converter
		converter=nbody_system.nbody_to_si(Mcluster,Rcluster)

		if W0==0.:
		    stars=new_plummer_sphere(N,converter)
		else:
		    stars=new_king_model(N,W0,convert_nbody=converter)

		if imf=='kroupa':
			if mmax <= 0.5 | units.MSun:
				stars=new_powerlaw_mass_distribution(number_of_particles=N,mass_min=mmin,mass_max=mmax,alpha=-1.3)

			elif mmin >=0.5 | units.MSun:
				stars.mass=new_powerlaw_mass_distribution(number_of_particles=N,mass_min=mmin,mass_max=mmax,alpha=-2.3)

			else:
				stars.mass=new_broken_power_law_mass_distribution(N,mass_boundaries= [mmin.value_in(units.MSun), 0.5, mmax.value_in(units.MSun)] | units.MSun,alphas= [-1.3,-2.3],mass_max=mmax )

		elif imf=='salpeter':
			stars.mass=new_powerlaw_mass_distribution(number_of_particles=N,mass_min=mmin,mass_max=mmax,alpha=alpha)

		elif imf=='single':
			stars.mass=np.ones(len(stars))*mmin

		    
	stars.scale_to_standard(convert_nbody=converter, smoothing_length_squared = softening2)
	stars.move_to_center()

	return stars,converter

