"""
The below script is an example of how to use m2mcluster to fit an M2M model to obserations of a Galactic
globuar cluster's density profile (m4_sig_inner_prof.dat, m4_sig_outer_prof.dat) and
kinematic properties (m4_pm_prof.dat, m4_rv_prof.dat). In this example, the M2M model is fit against
M4's inner and outer surface density profile, proper motion velocity dispersion, and line of sight 
velocity dipsersion. 

The initial model cluster is based on previous fits to M4.

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
restartsnap=0 #automatically looks for %s.csv % str(restartnap).zfill(5) to restart from
#****************************************


#**********Made to Measure Options***************

#Set Kernel type and M2M parameters
kernel=['loggaussian','gaussian','gaussian','gaussian']

epsilon= 10.0
mu=0.01
alpha=0.001

#Set limiting variables for reinitialization
rmax = 54.43 | units.parsec
mmin=0.1 | units.MSun
mmax=2.0 | units.MSun

#Set number of iterations and output frequency
niterations=10000
snapfreq=100
nextsnap=0

#**********Set observables**********
rhoparam=['Sigma1','Sigma2']
ndim=2

vfit=True
vparam=['v2','vz2']

#**********Nbody Simulations Options***************
#Need softening length and fraction of dynamical time that will be used for time steps
softening=0.01 | units.parsec
tdynrat=0.001

#**********Initial Particle Datasets*****
ofiles=['m4_sig_inner_prof.dat','m4_sig_outer_prof.dat','m4_pm_prof.dat','m4_rv_prof.dat']

#if intitial model star particles are in a file:
omodname='init_mod.dat'

#****************************************

#Initialize an M2M Star cluster
cluster=m2m.starcluster(number_of_iterations=niterations)

#Get Observables

for i,of in enumerate(ofiles):
    orlower,orad,orupper,o,eo=np.loadtxt(of,unpack=True)

    if '_sig_' in of:
        cluster.add_observable(orlower,orad,orupper,o,rhoparam[i],ndim=2,sigma=eo,kernel=kernel[i])
    elif '_pm_' in of and vfit:
        cluster.add_observable(orlower,orad,orupper,o,'v2',ndim=2,sigma=eo,kernel=kernel[i])
    elif '_rv_' in of and vfit:
        cluster.add_observable(orlower,orad,orupper,o,'vz2',ndim=2,sigma=eo,kernel=kernel[i])

#Initialize a model star cluster with an initial guess close to the observed cluster's properties
if not restart:
    cluster.initialize_star_cluster(filename=omodname, softening=softening)
elif restart:
    cluster.restart_star_cluster(restartsnap,1.,'m2moutfile.dat',softening=softening,unit='msunpckms',fmt='dwdt')
    nextsnap=restartsnap+snapfreq

#Write out initial conditions if not a restart
if not restart:
    cluster.writeout()
    cluster.snapout(return_dwdt=True)
    nextsnap+=snapfreq

#Calculate initial lvirial radius
r_v=cluster.stars.virial_radius()

#Remove stars outside of parameter space if a restart
if restart:
    cluster.reinitialize_star_cluster(mmin= mmin, mmax=mmax, rmax=rmax,rv=r_v)

#Calculate dynamical time for timestep calculation
tdyn=cluster.stars.dynamical_timescale()

#Plot initial comparisons
#Plot initial positions
cluster.xy_plot(filename='xyplot0.png')
#Compare initial density profiles
cluster.rho_prof(filename='rhoplot0.png')
#Compare initial velocity dispersion profiles
cluster.v2_prof(filename='vplot0.png')
cluster.rhov2_prof(filename='rhovplot0.png')

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
    
    #Centre the star cluster and find determine Nbody conversion scales for next integration
    cluster.reinitialize_star_cluster(mmin= mmin, mmax=mmax, rmax=rmax,rv=r_v)

    #Write profiles and chi^2 to outfile
    cluster.writeout()

    #Writeout snapshots at a given frequency and update virial radius
    if cluster.niteration>=nextsnap:
        cluster.snapout(return_dwdt=True)
        nextsnap+=snapfreq
        r_v=cluster.stars.virial_radius()

    sys.stdout.flush()
 
cluster.outfile.close()
cluster.gravity_code.stop()

#Plot final comparisons
#Plot final positions
cluster.xy_plot(filename='xyplotf.png')
#Compare final density profiles
cluster.rho_prof(filename='rhoplotf.png')
#Compare final velocity dispersion profiles
cluster.v_prof(filename='vplotf.png')



