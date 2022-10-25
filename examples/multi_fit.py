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
#Read in profiles directly
doprof=True
#Use a galpy potential instead of an Nbody simulations
dogalpy=False
#Restart
restart=False
restartsnap=0 #automatically looks for %s.csv % str(restartnap).zfill(5) to restart from
#****************************************


#**********Made to Measure Options***************

#Set Kernel type and M2M parameters
kernel='gaussian'

epsilon= 0.001
mu=0.0
alpha=0.0

#Set limiting variables for reinitialization
rmax = 54.43 | units.parsec
mmin=0.1 | units.MSun
mmax=2.0 | units.MSun
mtot = None

#Set number of iterations and output frequency
niterations=5103
snapfreq=100
nextsnap=0

#**********Set observables**********
rhoparam='Sigma'
ndim=2

vfit=True
vparam=['v2','vz2']

#If extend_out=True, then assume upper limit of outermost bins is infinity
extend_outer=False

#**********Nbody Simulations Options***************
#Need softening length and fraction of dynamical time that will be used for time steps
softening=0.01 | units.parsec
tdynrat=0.001

#If dogalpy, set the conversion factors
if dogalpy:
    from galpy.potential import KingPotential
    from galpy.util import conversion
    ro,vo=8.,220.
    mo=conversion.mass_in_msol(ro=ro,vo=vo)
    pot=KingPotential(M=Mcluster.value_in(units.MSun)/mo,rt=rmax/ro/1000.,ro=ro,vo=vo)

#**********Debugging***************
timing=True
doplots=False
debug=True


#**********Initial Particle Datasets*****

#if do prof:
ofiles=['init_sig_prof.dat','init_pm_prof.dat','init_rv_prof.dat']

#if initial observed star particles are in a file:
ofilename='init_obs.dat'

#if intitial model star particles are in a file:
omodname='init_mod.dat'

#If initial conditions are to be generated
#Observations
No=10000
Mclustero = 10000 | units.MSun
Rclustero = 10. |units.parsec
W0o=1.

#Model
N=10000
Mcluster = 10000 | units.MSun
Rcluster = 9. |units.parsec
W0=2.
#****************************************


#Initialize an M2M Star cluster
#Specify number of iterations to run algorithm for
#Specify number of workers to be used by Nbody code
cluster=m2m.starcluster(number_of_iterations=niterations,number_of_workers=1,debug=debug)

#Get Observables

if not doprof:
    if os.path.isfile(ofilename):
        ocluster,oconverter=m2m.setup_star_cluster(filename=ofilename)    
        #Measure artifical cluster's density profile asssuming fixed bins
        orlower,orad,orupper,orho=m2m.density(ocluster,param=rhoparam,ndim=ndim,bins=True,bintype='fix',kernel=kernel)
        #Manually set inner bin to cluster centre
        orlower[0]=0.

        if debug: print('DEBUG:',orlower,orad,orupper,orho)
                
    else:
        #Setup a star cluster for artificial observation
        ocluster,oconverter=m2m.setup_star_cluster(N=No, Mcluster=Mclustero, Rcluster = Rclustero, W0=W0o,imf='single',mmin=1. | units.MSun, mmax=1. | units.MSun)

        #Measure artifical cluster's density profile asssuming an equal number of stars per bin
        orlower,orad,orupper,orho=m2m.density(ocluster,param=rhoparam,ndim=ndim,bins=True,bintype='fix',kernel=kernel)
        orlower[0]=0.

        #Keep setting up an observed cluster until there are no bins with zero stars
        while np.sum(orho==0.)!=0:
            ocluster,oconverter=m2m.setup_star_cluster(N=No, Mcluster=Mclustero, Rcluster = Rclustero, W0=W0o,imf='single',mmin=1. | units.MSun, mmax=1. | units.MSun)
            orlower,orad,orupper,orho=m2m.density(ocluster,param=rhoparam,ndim=ndim,bins=True,bintype='fix',kernel=kernel)
            orlower[0]=0.

    #Find mean squared velocity in each bin if fitting against kinematics
    v=[]

    for vp in vparam:
        if 'rho' in vp or 'Sigma' in vp:
            v.append(m2m.density_weighted_mean_squared_velocity(ocluster,rlower=orlower,rmid=orad,rupper=orupper,param=vp,ndim=ndim,kernel=kernel))
        else:
            v.append(m2m.mean_squared_velocity(ocluster,rlower=orlower,rmid=orad,rupper=orupper,param=vp,ndim=ndim,kernel=kernel))


    #Add the "observed" cluster density profile as an observable
    cluster.add_observable(orlower,orad,orupper,orho,rhoparam,ndim=ndim,sigma=0.1*orho,kernel=kernel,extend_outer=extend_outer)

    #Add the "observed" cluster kinematic profiles as an observable
    if vfit:
        for i in range(0,len(vparam)):
            cluster.add_observable(orlower,orad,orupper,v[i],vparam[i],ndim=ndim,sigma=0.1*v[i],kernel=kernel,extend_outer=extend_outer)

elif doprof:

    for of in ofiles:
        orlower,orad,orupper,o,eo=np.loadtxt(of,unpack=True)

        if 'sig' in of:
            cluster.add_observable(orlower,orad,orupper,o,'Sigma',ndim=2,sigma=eo,kernel=kernel,extend_outer=extend_outer)
        elif 'pm' in of:
            cluster.add_observable(orlower,orad,orupper,o,'v2',ndim=2,sigma=eo,kernel=kernel,extend_outer=extend_outer)
        elif 'rv' in of:
            cluster.add_observable(orlower,orad,orupper,o,'vz2',ndim=2,sigma=eo,kernel=kernel,extend_outer=extend_outer)

#Initialize a model star cluster with an initial guess close to the observed cluster's properties
if os.path.isfile(omodname) and not restart:
    cluster.initialize_star_cluster(filename=omodname, softening=softening)
elif not restart:
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
if debug: print('START VIRIAL RADIUS',cluster.stars.total_mass().value_in(units.MSun))
r_v=cluster.stars.virial_radius()
if debug: print('VIRIAL RADIUS = ',r_v.value_in(units.parsec))

#Remove stars outside of parameter space if a restart
if restart:
    if debug: print('REINITIALIZE: ',np.sum(cluster.stars.mass.value_in(units.MSun)<mmin.value_in(units.MSun)),np.sum(cluster.stars.mass.value_in(units.MSun)>mmax.value_in(units.MSun)))
    cluster.reinitialize_star_cluster(mmin= mmin, mmax=mmax, mtot=mtot, rmax=rmax,rv=r_v)

#Calculate dynamical time for timestep calculation
tdyn=cluster.stars.dynamical_timescale()
if debug: print(tdyn.value_in(units.Myr),cluster.tdyn.value_in(units.Myr))


#Make initial plot
if doplots: 
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
    if timing: dttime=time.time()
    #Initialize a new N-body simulation ever time step. 
    #In this example I use 1% of the cluster's dynamical time for the integration timeste
    cluster.gravity_code=None
    if dogalpy:
        cluster.initialize_gravity_code('Galpy', dt=tdynrat*cluster.tdyn, theta=0.6, pot=pot)
    else:
        cluster.initialize_gravity_code('BHTree', dt=tdynrat*cluster.tdyn, theta=0.6)

    if timing: print('initialize_gravity_code',time.time()-dttime)
    if timing: dttime=time.time()
    
    #Evolve the model cluster forward for tdynrat percent of its dynamical time
    #tnext=tdynrat*cluster.tdyn*(cluster.niteration+1-restartsnap)
    tnext=tdynrat*cluster.tdyn
    if debug: print('TDYN: ',tdynrat,cluster.tdyn)
    if debug: print('MOD1: ',cluster.models['Sigma'])
    cluster.evolve(tend=tnext)
    cluster.gravity_code.stop() 
    if debug: print('MOD2: ',cluster.models['Sigma'])

    if timing: print('evolve',time.time()-dttime)
    if timing: dttime=time.time()
    
    #Run the M2M algorithm, which will adjust all of the stellar masses based on kernel function
    cluster.evaluate(epsilon=epsilon,mu=mu,alpha=alpha)
    if debug: print('MOD3: ',cluster.models['Sigma'])
    #cluster.niteration+=1 
    if timing: print('evaluate',time.time()-dttime)
    
    #Centre the star cluster and find determine Nbody conversion scales for next integration
    if debug: print('REINITIALIZE: ',np.sum(cluster.stars.mass.value_in(units.MSun)<mmin.value_in(units.MSun)),np.sum(cluster.stars.mass.value_in(units.MSun)>mmax.value_in(units.MSun)))
    if timing: dttime=time.time()
    cluster.reinitialize_star_cluster(mmin= mmin, mmax=mmax, mtot=mtot, rmax=rmax,rv=r_v)

    if timing: print('reinitialize_star_cluster',time.time()-dttime)
    if timing: dttime=time.time()

    #Write profiles and chi^2 to outfile
    cluster.writeout()
    
    if timing: print('writeout',time.time()-dttime)
    if timing and doplots: dttime=time.time()
        
    #Compare the new model density profile to the observed one
    if doplots: 
        cluster.rho_prof(filename='%s.png' % str(i).zfill(5))
        cluster.v2_prof(filename='%s_v.png' % str(i).zfill(5))
        cluster.rhov2_prof(filename='%s_rhov.png' % str(i).zfill(5))

    if timing and doplots: print('rho_prof',time.time()-dttime)
    if timing: dttime=time.time()

    #Writeout snapshots at a given frequency and update virial radius
    if cluster.niteration>=nextsnap:
        cluster.snapout(return_dwdt=True)
        nextsnap+=snapfreq
        if debug: print('START VIRIAL RADIUS',cluster.stars.total_mass().value_in(units.MSun))
        r_v=cluster.stars.virial_radius()
        if debug: print('VIRIAL RADIUS = ',r_v,nextsnap,nextsnap%100)

        if timing: print('snapout',time.time()-dttime)

    sys.stdout.flush()
 
cluster.outfile.close()

if cluster.bridge:
    cluster.gravity_bridge.stop()
else:
    cluster.gravity_code.stop()


#Make final comparisons
if doplots: 
    #Plot initial positions
    cluster.xy_plot(filename='xyplotf.png')
    #Compare initial density profiles
    cluster.rho_prof(filename='rhoplotf.png')
    #Compare initial velocity dispersion profiles
    cluster.v_prof(filename='vplotf.png')



