"""
The below script is an example of how to use m2mcluster to fit an M2M model to a previously genereated
mock isotropic dataset (mock_cluster.dat). In this example, only the mock clusters three dimensional
density profile is being used as the target observable

The initial model cluster is purposefully assumed to be quite different than the target cluster to
demonstrate the stength of the M2M approach.

See Webb, Hunt, and Bovy 2023 for details

"""

import numpy as np
import os,sys
import time

#Import m2mcluster
import m2mcluster as m2m

#Import relevant Amuse module
from amuse.lab import *
from amuse.units import nbody_system,units
from amuse.datamodel import Particles

#**********Initial Options***************
#Restart
restart=False
restartsnap=0 
#****************************************

#**********Made to Measure Options***************

#Set Kernel type and M2M parameters
kernel='gaussian'

epsilon= 0.005
mu=0.005
alpha=0.0

#Set limiting variables for reinitialization
rmax = 24.1 | units.parsec
mmin=0.1 | units.MSun
mmax=2.0 | units.MSun

#Set number of iterations and output frequency
niterations=100
snapfreq=20
nextsnap=0

#**********Set observables**********
#Target density parameter ('rho' or 'Sigma') and dimensionality for radii (2 or 3)
rhoparam='rho'
ndim=3

#Target kinematic parameters ('rhov2','vr2','vp2','vt2','vR2','vlos2','vR2',vT2','vz2','v2')
vfit=False
vparam=None


#**********Nbody Simulations Options***************
#Need softening length and fraction of dynamical time that will be used for time steps
softening=0.01 | units.parsec
tdynrat=0.001


#**********Initial Particle Datasets*****

#Filename of mock observed star particles:
ofilename='mock_cluster.dat'

#Initial conditions of M2M model cluster, to be generated as a King Model with AMUSE
N=10000
Mcluster = 10000 | units.MSun
Rcluster = 9. |units.parsec
W0=2.
#****************************************


#Initialize an M2M Star cluster
cluster=m2m.starcluster(number_of_iterations=niterations)

#Measure Observable Parameter in Mock Observed Cluster
ocluster,oconverter=m2m.setup_star_cluster(filename=ofilename)    
#Measure artifical cluster's density profile asssuming fixed bins
orlower,orad,orupper,orho=m2m.density(ocluster,param=rhoparam,ndim=ndim,bins=True,bintype='fix',kernel=kernel)
#Manually set inner bin to cluster centre
orlower[0]=0.

#Add the "observed" cluster density profile as an observable
#Note for this example the uncertainty is density is assumed to be 10 percent (sigma=0.1*orho)
cluster.add_observable(orlower,orad,orupper,orho,rhoparam,ndim=ndim,sigma=0.1*orho,kernel=kernel)

#Initialize a model star cluster with an initial guess close to the observed cluster's properties
if not restart:
    cluster.initialize_star_cluster(N=N, Mcluster = Mcluster, Rcluster = Rcluster, softening = softening, W0=W0,imf='single',mmin=1. | units.MSun)
elif restart:
    cluster.restart_star_cluster(restartsnap,1.,'m2moutfile.dat',softening=softening,unit='msunpckms',fmt='new')
    nextsnap=restartsnap+snapfreq

#Write out initial conditions if not a restart
if not restart:
    cluster.writeout()
    cluster.snapout()
    nextsnap+=snapfreq


#Calculate initial lvirial radius
r_v=cluster.stars.virial_radius()

#Remove stars outside of parameter space if a restart
if restart:
    cluster.reinitialize_star_cluster(mmin= mmin, mmax=mmax, rmax=rmax,rv=r_v)

#Calculate dynamical time for timestep calculation
tdyn=cluster.stars.dynamical_timescale()


#Plot initial comparisons
cluster.xy_plot(filename='xyplot0.png')
#Compare initial density profiles
cluster.rho_prof(filename='rhoplot0.png')

#Exectute the made to measure algorithm
for i in range(restartsnap,cluster.number_of_iterations):

    print(cluster.niteration,restartsnap,cluster.tdyn.value_in(units.Myr),tdynrat)

    #Initialize a new N-body simulation ever time step. 
    #In this example I use 1% of the cluster's dynamical time for the integration timeste
    cluster.gravity_code=None
    cluster.initialize_gravity_code('BHTree', dt=tdynrat*cluster.tdyn, theta=0.6)
    
    #Evolve the model cluster forward for tdynrat percent of its dynamical time
    tnext=tdynrat*cluster.tdyn
    cluster.evolve(tend=tnext)
    cluster.gravity_code.stop() 
    
    #Run the M2M algorithm, which will adjust all of the stellar masses based on kernel function
    cluster.evaluate(epsilon=epsilon,mu=mu,alpha=alpha)
    
    #Write profiles and chi^2 to outfile
    cluster.writeout()

    #Writeout snapshots at a given frequency and update virial radius
    if cluster.niteration>=nextsnap:
        cluster.snapout()
        nextsnap+=snapfreq
        r_v=cluster.stars.virial_radius()

    #Centre the star cluster and find determine Nbody conversion scales for next integration
    cluster.reinitialize_star_cluster(mmin= mmin, mmax=mmax, rmax=rmax,rv=r_v)

    sys.stdout.flush()
 
cluster.outfile.close()
cluster.gravity_code.stop()

#Plot final comparisons
cluster.xy_plot(filename='xyplotf.png')
#Compare final density profiles
cluster.rho_prof(filename='rhoplotf.png')




