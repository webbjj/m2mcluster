#import matplotlib
#matplotlib.use('Agg')

from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.units import nbody_system, units

from .functions import *
from .plot import *
from .kernels import *
import numpy
import matplotlib.pyplot as pyplot
from scipy.stats import chisquare

from amuse.ext.LagrangianRadii import LagrangianRadii

def made_to_measure(stars,observed_rho,observed_sigv,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,delta_j_tilde=None,kernel=None,debug=False,plot=False,filename=None,**kwargs,):
    
    #Get the observed diensity profile
    rlower,rmid,rupper,rho, param, ndim=observed_rho

    #Calculate volume of bin shells
    vol=(4./3.)*numpy.pi*(rupper**3.-rlower**3.)

    #Get the model cluster's current density using the same radial bins as the observed density profile
    mod_rho=density(stars,rlower,rmid,rupper,ndim)

    if plot:
        density_profile(stars,observation,filename=filename)

    #Entropy:
    #S=-mu*np.sum(stars.mass.value_in(units.MSun)*np.log(stars.mass.value_in(units.MSun)/w0.value_in(units.MSun)-1.))
    dsdw=-mu*np.log(stars.mass.value_in(units.MSun)/w0)

    #Calculate delta_j (yj/Yj-1)
    delta_j=mod_rho/rho-1.
    Y_j=rho

    #D. Syer & S. Tremaine 1996 - Equation 15
    if delta_j_tilde is None:
        delta_j_tilde=delta_j
    else:
        d_delta_j_tilde_dt=alpha*(delta_j-delta_j_tilde)
        delta_j_tilde=delta_j-d_delta_j_tilde_dt/alpha


    if debug: print('OBS_RHO', rho)
    if debug: print('MOD_RHO', mod_rho)
    if debug: print('DELTA = ',delta_j)
    if debug: print('DELTA_TILDE: ',delta_j_tilde)

    #Initialize Gaussian kernel if called (WIP)
    if kernel == 'gaussian':
        #D. Syer & S. Tremaine 1996 - Section 2.2 - use bin centre as mean of Gaussian and sigma of 1/2 bin width
        sigma=kwargs.get('sigma',(rupper-rlower)/2.)

    r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)

    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Find mean dwdt in each bin for debugging
    if debug:
        dwdtr=np.zeros(len(rmid))
        ndwdtr=np.zeros(len(rmid))

    #Find dwdt for each star
    for i in range(0,len(stars)):

        if kernel is None:
            rindx=((r[i] >= rlower) * (r[i] <= rupper))

            #D. Syer & S. Tremaine 1996 - Section 2.2, set K_j to 1/vol or 0, such that K_j/Z_j = 1 or 0
            #Similarly K_j/Y_j will be the number of stars in the bin 
            K_j=rindx*(1./vol) 


        elif kernel == 'gaussian':

            rindx=((r[i] >= rlower) * (r[i] <= rupper))

            if np.sum(rindx)!=0:
                rmean=rmid[rindx][0]
                sig=sigma[rindx][0]

                K_j=gaussian_kernel(rmean,rmean,sig)
            else:
                K_j=np.zeros(len(rlower))

            K_j=K_j*(1./vol) 


        #D. Syer & S. Tremaine 1996 - Equation 4
        dwdt[i]=epsilon*stars[i].mass.value_in(units.MSun)*(mu*dsdw[i]-numpy.sum(K_j*delta_j_tilde/Y_j))
        

        stars[i].mass += dwdt[i] | units.MSun

        if debug:
            dwdtr[rindx]+=dwdt[i]
            ndwdtr[rindx]+=1.

    #Goodness of fit criteria - D. Syer & S. Tremaine 1996 - Equation 22
    chi_squared=chi2(mod_rho,rho)

    if debug: print('WEIGHTING: ',np.amin(dwdt),np.mean(dwdt),np.amax(dwdt),np.amin(dsdw),np.mean(dsdw),np.amax(dsdw),chi_squared)
    if debug: print('MEAN DWDT: ',dwdtr[ndwdtr>0]/ndwdtr[ndwdtr>0])


    return stars,chi_squared,delta_j_tilde
