import m2m_cluster,m2m_algorithm
import m2m_functions,m2m_plot

from amuse.lab import *
from amuse.units import nbody_system,units
from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.ic.plummer import new_plummer_model


from amuse.community.bhtree.interface import BHTree

import numpy

#Output a snapshot
def output(cluster,filename):
    numpy.savetxt(filename,numpy.column_stack([cluster.mass.value_in(units.MSun),cluster.x.value_in(units.parsec),cluster.y.value_in(units.parsec),cluster.z.value_in(units.parsec),cluster.vx.value_in(units.kms),cluster.vy.value_in(units.kms),cluster.vz.value_in(units.kms)]))

#Read in a snapshot
def input(filename):
    data=numpy.loadtxt(filename)
    cluster=Particles(len(data))
    cluster.mass=data[:,0] | units.MSun
    cluster.x=data[:,1] | units.kpc
    cluster.y=data[:,2] | units.kpc
    cluster.z=data[:,3] | units.kpc
    cluster.vx=data[:,4] | units.kms
    cluster.vy=data[:,5] | units.kms
    cluster.vz=data[:,6] | units.kms

    return cluster


do_cluster_only=False

number_of_stars=100000
Mcluster=60000.0 | units.MSun
Rcluster=3.0 | units.parsec

epsilon2=(0.75 | units.parsec)**2
kmax=1.965784284662087 #2^-8 Gyr = 2.0**1.96 Myr
Nlev=3.
theta=0.6

#Set how many iterations to run M2M for (or create a criteria)
m2m_iterations=1000
n_iteration=0


#Setup initial model and observed clusters

if False:
    obs_particles=input('init.dat')
    Mcluster=obs_particles.total_mass()
    Rcluster=obs_particles.virial_radius()
    obs_converter=nbody_system.nbody_to_si(Mcluster,Rcluster)

    mod_particles=input('init.dat')
    Mcluster=mod_particles.total_mass()
    Rcluster=mod_particles.virial_radius()
    mod_converter=nbody_system.nbody_to_si(Mcluster,Rcluster)

else:
    obs_particles,obs_converter=m2m_cluster.create_cluster(N=number_of_stars,Mcluster=Mcluster,Rcluster=Rcluster,epsilon2=epsilon2,W0=1.,imf='kroupa')
    mod_particles,mod_converter = m2m_cluster.create_cluster(N=number_of_stars,Mcluster=Mcluster,Rcluster=3.*Rcluster,epsilon2=epsilon2,W0=7.,imf='kroupa')


tend=0.1*m2m_functions.get_dynamical_time_scale(Mcluster, Rcluster)

#Find density in grid based on lagrange radii
lr=m2m_functions.get_lagrange_radii(obs_particles)
obs_rlower,obs_rad,obs_rupper,obs_rho=m2m_functions.density(obs_particles,lr,n_iteration,'obs')
    
#Find density in grid based on observed lagrange radii
mod_rlower,mod_rad,mod_rupper,mod_rho=m2m_functions.density(mod_particles,lr,n_iteration,'mod')

while n_iteration<=m2m_iterations:
    print(n_iteration, ' TIME UNITS: ',tend.as_quantity_in(units.Myr))

    #output(mod_particles,str(n_iteration).zfill(5)+'.initial.dat')

    if do_cluster_only:
        m2m_plot.plot_positions(mod_particles.x,mod_particles.y,mod_particles.z,n_iteration,'mod')

        cluster_code=BHTree(mod_converter)
        cluster_code.parameters.epsilon_squared = epsilon2
        cluster_code.parameters.opening_angle=theta
        cluster_code.parameters.timestep=numpy.minimum(cluster_code.parameters.timestep,tend/(2.0**(Nlev-1.)))
        cluster_code.particles.add_particles(mod_particles)
        
        channel_from_stars_to_cluster=mod_particles.new_channel_to(cluster_code.particles, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])
        
        channel_from_cluster_to_stars=cluster_code.particles.new_channel_to(mod_particles, attributes=["mass", "x", "y", "z", "vx", "vy", "vz"])
        cluster_code.evolve_model(tend)
        time = cluster_code.model_time
        print('CLUSTER EVOLVED TO: ',time.as_quantity_in(units.Myr))
        channel_from_cluster_to_stars.copy()
        cluster_code.stop()


    else:
        #Evolve cluster
        mod_particles = m2m_cluster.evolve_cluster(mod_particles,mod_converter,tend,epsilon2,kmax,Nlev,theta)
        mod_particles.move_to_center()

        m2m_plot.plot_positions(mod_particles.x,mod_particles.y,mod_particles.z,n_iteration,'mod')


        #Run made to measure algorithm and get new masses
        mod_partices,m2m_criteria=m2m_algorithm.made_to_measure(mod_particles,obs_particles,n_iteration)

        print('DELTA M: ',Mcluster.value_in(units.MSun),mod_particles.total_mass().value_in(units.MSun))
        print('DELTA R: ',Rcluster.value_in(units.parsec),mod_particles.virial_radius().value_in(units.parsec))

        #Reinitialize cluster
        mod_particles.move_to_center()
        Mcluster=mod_particles.total_mass()
        Rcluster=mod_particles.virial_radius()
        mod_converter=nbody_system.nbody_to_si(Mcluster,Rcluster)
        tend=0.1*m2m_functions.get_dynamical_time_scale(Mcluster, Rcluster)
        print(m2m_criteria,n_iteration)

    n_iteration+=1

