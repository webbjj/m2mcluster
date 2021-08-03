#import matplotlib
#matplotlib.use('Agg')

from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.units import nbody_system, units

from .functions import *
from .plot import *
from .kernels import *
import numpy
from scipy.stats import chisquare

from amuse.ext.LagrangianRadii import LagrangianRadii

def made_to_measure(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,delta_j_tilde=None,kernel=None,rhov2=False,method='Seyer',debug=False,**kwargs,):
    
    #JO - alpha >= epsilon?
    #JO - alpha=1/dt is equivalent to no smoothing and alpha cannot be set to a larger value. Which dt? integartion time or eval time?

    if len(observations)==1 and ('rho' in observations or 'Sigma' in observations) or method=='Seyer':

        stars,chi_squared,delta_j_tilde=made_to_measure_seyer(stars,observations,w0,epsilon,mu,alpha,delta_j_tilde,kernel,debug,**kwargs,)
        return stars,chi_squared,delta_j_tilde

    elif len(observations)==2 and ('rho' in observations or 'Sigma' in observations) and ('v2' in observations or 'vlos2' in observations or 'vR2' in observations or 'vT2' in observations or 'vz2' in observations) and rhov2:

        stars,chi_squared,delta_j_tilde,delta_j2_tilde=made_to_measure_bovy(stars,observations,w0,epsilon,mu,alpha,delta_j_tilde,kernel,rhov2,debug,**kwargs,)
        return stars,chi_squared,delta_j_tilde,delta_j2_tilde


def made_to_measure_seyer(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,delta_j_tilde=None,kernel=None,debug=False,**kwargs,):

    #Get the observed diensity profile
    if 'rho' in observations:
        rlower,rmid,rupper,rho, param, ndim, sigma=observations['rho']
    else:
        rlower,rmid,rupper,rho, param, ndim, sigma=observations['Sigma']

    #Get the model cluster's current density using the same radial bins as the observed density profile
    mod_rho=density(stars,rlower,rmid,rupper,param,ndim)

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

    if ndim==3:
        r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
    elif ndim==2:
        r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)

    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Find dwdt for each star
    for i in range(0,len(stars)):

        K_j=get_kernel(r[i],rlower,rmid,rupper,kernel,**kwargs)
        dchi2=get_dchi2(delta_j_tilde,K_j,Y_j)

        #D. Syer & S. Tremaine 1996 - Equation 4
        dwdt[i]=epsilon*stars[i].mass.value_in(units.MSun)*(dsdw[i]-numpy.sum(dchi2)/2.)
        
        stars[i].mass += dwdt[i] | units.MSun

    #Goodness of fit criteria - D. Syer & S. Tremaine 1996 - Equation 22
    chi_squared=np.sum((delta_j)**2.)/len(delta_j)

    return stars,chi_squared,delta_j_tilde

def made_to_measure_bovy(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,delta_j0_tilde=None,kernel=None,rhov2=True,debug=False,**kwargs,):
    
    #Get the observed diensity profile

    rlowers=[]
    rmids=[]
    ruppers=[]
    obs=[]
    params=[]
    ndims=[]
    sigmas=[]

    for oparam in self.observations:

        rlower,rmid,rupper,ob,param,ndim,sigma=observations[oparam]
        rlowers.append(rlower)
        rmids.append(rmid)
        ruppers.append(rupper)
        obs.apend(ob)
        params.apend(param)
        ndims.apend(ndim)
        sigmas.apend(sigma)

    mods=[]

    for i in range(0,len(obs)):
        if params[i]=='rho' or params[i]=='Sigma':
            mod=density(stars,rlowers[i],rmids[i],ruppers[i],params[i],ndims[i])
        else:
            mod=mean_squared_velocity(stars,rlowers[i],rmids[i],ruppers[i],params[i],ndims[i])
        mods.append(mod)

    #Entropy:Bovy Equation 21
    #S=-mu*np.sum(stars.mass.value_in(units.MSun)*np.log(stars.mass.value_in(units.MSun)/w0.value_in(units.MSun)-1.))
    dsdw=-mu*np.log(stars.mass.value_in(units.MSun)/w0)

    #Calculate delta_j
    delta_j=[]
    mod_rho,rho,mod_v2,v2=None,None,None,None
    for i in range(0,len(obs)):

        if (pararms[i]=='rho' or params[i]=='Sigma') and rhov2:
            mod_rho=mods[i]
            rho=obs[i]
            delta_j.append(mods[i]-rho[i])

        elif (pararms[i]=='rho' or params[i]=='Sigma'):
            delta_j.append(mods[i]-rho[i])

        elif 'v' in params[i] and rhov2:
            mod_v2=mods[i]
            v2=obs[i]
        elif 'v' in params[i]:
            delta_j.append(mods[i]-rho[i])

        if rhov2 and (mod_rho is not None) and (mod_v2 is not None):
            delta_j.append(mod_rho*mod_v2-rho*v2)

    #Bovy Equation 22/23
    if delta_j_tilde is None:
        delta_j_tilde=delta_j
    else:
        for i in range(0,len(delta_j_tilde)):
            d_delta_j_tilde_dt=(alpha*(delta_j[i]-delta_j_tilde[i]))
            delta_j_tilde[i]=delta_j[i]-d_delta_j_tilde_dt/alpha

    if ndim==3:
        r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
        v2=(stars.vx.value_in(units.kms))**2.+(stars.vy.value_in(units.kms))**2.+(stars.vz.value_in(units.kms))**2.

    elif ndim==2:
        r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)
        v2=(stars.vx.value_in(units.kms))**2.+(stars.vy.value_in(units.kms))**2.
    elif ndim==1:
        r=numpy.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)
        v2=(stars.vx.value_in(units.kms))**2.    

    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Find dwdt for each star
    for i in range(0,len(stars)):

        K_j=[]

        for j in range(0,len(obs)):
            K_j.append(get_kernel(r[i],rlowers[j],rmids[j],ruppers[j],kernel=kernel,**kwargs))

        dchi2=[]

        for j in range(0,len(obs)):
            dchi2.append(get_dchi2(delta_j_tilde[j],K_j[j],sigmas[j],rhov2,mod_v2))

        #Bovy Equation 18
        dchisum=0

        for j in range(0,len(dchi2)):
            dchisum+=np.sum(dchi2[j])/2.

        dwdt[i]=epsilon*stars[i].mass.value_in(units.MSun)*(dsdw[i]-dchisum)
        
        stars[i].mass += dwdt[i] | units.MSun

    #Sum over Chi2 contributions
    chi_squared=0.
    for i in range(0,len(obs)):
        chi_squared+=np.sum((delta_j_tilde[i]/sigmas[i])**2.)

    return stars,chi_squared,delta_j_tilde,delta_j2_tilde

def get_kernel(r,rlower,rmid,rupper,kernel=None,*kwargs):

    vol=(4./3.)*numpy.pi*(rupper**3.-rlower**3.)

    if kernel=='gaussian':
        #D. Syer & S. Tremaine 1996 - Section 2.2 - use bin centre as mean of Gaussian and sigma of 1/2 bin width
        ksigma=kwargs.get('ksigma',(rupper-rlower)/2.)

    if kernel is None:
        rindx=((r >= rlower) * (r <= rupper))

        #D. Syer & S. Tremaine 1996 - Section 2.2, set K_j to 1/vol or 0, such that K_j/Z_j = 1 or 0
        K_j=rindx*(1./vol) 

    elif kernel == 'gaussian':
        rindx=((r >= rlower) * (r <= rupper))

        if np.sum(rindx)!=0:
            rmean=rmid[rindx][0]
            sig=ksigma[rindx][0]
            K_j=gaussian_kernel(rmean,rmean,sig)
        else:
            K_j=np.zeros(len(rlower))

        K_j=K_j*(1./vol)

    return K_j

def get_dchi2(delta_j_tilde,K_j,sigma_j,rhov2=False,v2=None):

    #From Bovy Equations 19,20, and A3
    if rhov2:
        dchi2=2.0*delta_j_tilde*v2*K_j/(sigma_j**2.)
    else:
        dchi2=2.0*delta_j_tilde*K_j/(sigma_j**2.)

    return dchi2








