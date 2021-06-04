#import matplotlib
#matplotlib.use('Agg')

from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.units import nbody_system, units

from .functions import *
from .plot import *
import numpy
import matplotlib.pyplot as pyplot
from scipy.stats import chisquare

from amuse.ext.LagrangianRadii import LagrangianRadii

def made_to_measure(stars,observation,n_iteration,kernel=None,m2mepsilon=10.0**6.,debug=False,plot=False,filename=None):
    
    rlower,rupper,rho, param, ndim=observation
    rad=(rlower+rupper)/2.
    vol=(4./3.)*numpy.pi*(rupper**3.-rlower**3.)
    area=numpy.pi*(rupper**2.-rlower**2.)


    mod_rho=density(stars,rlower,rupper,ndim)

    if plot:
        density_profile(stars,observation,filename=filename)

    #Goodness of fit criteria?
    chi_squared=chi2(mod_rho,rho)

    #Need to code M2M algorithm here
    delta=mod_rho/rho-1.
    Z_j=1.0/vol

    if debug: print('OBS_RHO', rho)
    if debug: print('MOD_RHO', mod_rho)
    if debug: print('DELTA = ',delta)
    
    r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
    #What do we choose for m2mepsilon - controls rate of change of wieght change (too large shows big fluctuations and too small barely moves)
    #10 resulted in density profile flying by control

    #m2mepsilon=0.
    #print('Z_J: ',Z_j)
    #Z_j=numpy.ones(len(Z_j))
    dw=[]

    #Zj is equal to inverse of volume of the bin
    #Look for near/max of 10% mass change
    for i in range(0,len(stars)):
        #print('BEFORE: ',i,mod_particles[i].mass.value_in(units.MSun),dwdt)

        K_j=((r[i] >= rlower) * (r[i] <= rupper))
        dwdt=-1.*m2mepsilon*stars[i].mass.value_in(units.MSun)*numpy.sum(K_j*delta/Z_j)
        dw.append(dwdt)

        stars[i].mass += dw[-1] | units.MSun

        if stars[i].mass < 0.0 | units.MSun:
            stars[i].mass = 0.0 | units.MSun

        #print('AFTER: ',i,mod_particles[i].mass.value_in(units.MSun),dwdt)
 
    if debug: print('WEIGHTING: ',delta,dw[0],dw[-1],chi_squared,m2mepsilon)



    return stars,chi_squared
