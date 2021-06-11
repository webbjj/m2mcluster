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

def made_to_measure(stars,observation,n_iteration,kernel=None,m2mepsilon=10.0**6.,debug=False,plot=False,filename=None,**kwargs):
    
    #Get the observed diensity profile
    rlower,rmid,rupper,rho, param, ndim=observation
    #Calculate volume of bin shells
    vol=(4./3.)*numpy.pi*(rupper**3.-rlower**3.)

    #Get the model cluster's current density using the same radial bins as the observed density profile
    mod_rho=density(stars,rlower,rmid,rupper,ndim)

    if plot:
        density_profile(stars,observation,filename=filename)

    #Calculate delta_j (yj/Yj-1)
    delta_j=mod_rho/rho-1.
    #Set z_j equal to inverse of volume of the bin
    Z_j=1.0/vol

    if debug: print('OBS_RHO', rho)
    if debug: print('MOD_RHO', mod_rho)
    if debug: print('DELTA = ',delta_j)
    if debug: print('Z_J: ',Z_j)

    #Initialize Gaussian kernel if called
    sigma=kwargs.get('sigma',None)
    if kernel == 'gaussian':
        #D. Syer & S. Tremaine 1996 - Section 2.2 - use bin centre as mean of Gaussian and sigma of 1/2 bin width
        if sigma is None:
            sigma=(rupper-rlower)/2.

    r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)


    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))


    #Find dwdt for each star
    for i in range(0,len(stars)):

        if kernel is None:
            K_j=((r[i] >= rlower) * (r[i] <= rupper))


            #JASON - If stars are not within any bin, match to closest bin or leave alone?
            if np.sum(K_j)==0 and r[i]>rupper[-1]:
                K_j[-1]=True
            elif np.sum(K_j)==0 and r[i]<rlower[0]:
                K_j[0]=True



            K_j=K_j*(1./vol) #D. Syer & S. Tremaine 1996 - Section 2.2, set K_j to 1/vol or 0, such that K_j/Z_j = 1 or 0


        elif kernel == 'gaussian':

            rindx=((r[i] >= rlower) * (r[i] <= rupper))

            if np.sum(rindx)==0 and r[i]>rupper[-1]:
                rmean=rmid[-1]
                sig=sigma[-1]
            elif np.sum(rindx)==0 and r[i]<rlower[0]:
                rmean=rmid[0]
                sig=sigma[0]
            else:
                rmean=rmid[rindx][0]
                sig=sigma[rindx][0]

            #JASON - confirm that I call the gaussian at the bin centre, not the star's position?
            K_j=gaussian_kernel(rmean,rmean,sig)

        if debug: print('DEBUG: ',K_j,delta_j,Z_j,K_j*delta_j)

        #D. Syer & S. Tremaine 1996 - Equation 4
        dwdt[i]=-1.*m2mepsilon*stars[i].mass.value_in(units.MSun)*numpy.sum(K_j*delta_j/Z_j)
        

        #JASON - Do I need to multiple by dt here? What units? Or does this kind of balance out with m2mepsilon?
        stars[i].mass += dwdt[i] | units.MSun

        if stars[i].mass < 0.0 | units.MSun:
            stars[i].mass = 0.0 | units.MSun

    #Goodness of fit criteria - D. Syer & S. Tremaine 1996 - Equation 22
    #JASON - I never do anything with this at the moment. Related to not including a penalty?
    #JASOn - check functional form with and without kernel
    chi_squared=chi2(mod_rho,rho,sigma)

    if debug: print('WEIGHTING: ',delta_j,dwdt[0],dwdt[i],chi_squared,m2mepsilon)



    return stars,chi_squared
