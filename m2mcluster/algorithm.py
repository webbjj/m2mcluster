#import matplotlib
#matplotlib.use('Agg')

from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.units import nbody_system, units

from .functions import *
from .plot import *
from .kernels import *
import numpy as np
from scipy.stats import chisquare

from amuse.ext.LagrangianRadii import LagrangianRadii

def made_to_measure(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,step=1.,delta_j_tilde=None,method='Seyer',debug=False,**kwargs,):
    
    #JO - alpha >= epsilon?
    #JO - alpha=1/dt is equivalent to no smoothing and alpha cannot be set to a larger value. Which dt? integartion time or eval time?

    if method=='Seyer':

        stars,chi_squared,delta_j_tilde=made_to_measure_seyer(stars,observations,w0,epsilon,mu,alpha,step,delta_j_tilde,debug,**kwargs,)
        return stars,chi_squared,delta_j_tilde

    else:

        stars,chi_squared,delta_j_tilde=made_to_measure_bovy(stars,observations,w0,epsilon,mu,alpha,step,delta_j_tilde,debug,**kwargs,)
        return stars,chi_squared,delta_j_tilde


def made_to_measure_seyer(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,step=1.,delta_j_tilde=None,debug=False,**kwargs,):

    #Get the observed density profile
    if 'rho' in observations:
        rlower,rmid,rupper,rho, param, ndim, sigma, rhokernel=observations['rho']
    else:
        rlower,rmid,rupper,rho, param, ndim, sigma, rhokernel=observations['Sigma']

    #Get the model cluster's current density using the same radial bins as the observed density profile
    mod_rho=density(stars,rlower,rmid,rupper,param,ndim,kernel=rhokernel,**kwargs)
    #Entropy:
    #S=-mu*np.sum(stars.mass.value_in(units.MSun)*np.log(stars.mass.value_in(units.MSun)/w0.value_in(units.MSun)-1.))
    dsdw=-mu*np.log(stars.mass.value_in(units.MSun)/w0)

    #Calculate delta_j (yj/Yj-1)
    delta_j=mod_rho/rho-1.
    Y_j=rho

    #D. Syer & S. Tremaine 1996 - Equation 15
    if delta_j_tilde is None or alpha==0.:
        delta_j_tilde=delta_j
    else:
        delta_j_tilde+=step*alpha*(delta_j-delta_j_tilde)

    if debug: 
        print('OBS_RHO', rho)
        print('MOD_RHO', mod_rho)
        print('DELTA = ',delta_j)
        print('DELTA_TILDE: ',delta_j_tilde)

    if ndim==3:
        r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
    elif ndim==2:
        r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)

    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Find dwdt for each star
    for i in range(0,len(stars)):

        K_j=get_kernel(r[i],rlower,rmid,rupper,rhokernel,ndim,**kwargs)
        #Note the sqrt on Y_j due to get_dchi2 taking sigma in bovy formalism
        dchi2=get_dchi2(delta_j_tilde,K_j,np.sqrt(Y_j))

        #D. Syer & S. Tremaine 1996 - Equation 4
        dwdt[i]=epsilon*stars[i].mass.value_in(units.MSun)*(dsdw[i]-np.sum(dchi2)/2.)

        stars[i].mass += step*dwdt[i] | units.MSun

        if r[i]<rupper[0] and debug:
            print('DWDT INNERMOST: ',epsilon,stars[i].mass.value_in(units.MSun),dsdw[i],dchi2[dchi2!=0.],(dsdw[i]-np.sum(dchi2)/2.),dwdt[i])


    #Goodness of fit criteria - D. Syer & S. Tremaine 1996 - Equation 22
    chi_squared=np.sum((delta_j)**2.)/len(delta_j)

    if debug: 
        print('WEIGHTING: ',np.amin(dwdt),np.mean(dwdt),np.amax(dwdt),np.amin(dsdw),np.mean(dsdw),np.amax(dsdw),chi_squared)

    return stars,chi_squared,delta_j_tilde

def made_to_measure_bovy(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,step=1.,delta_j_tilde=None,debug=False,**kwargs,):
    
    #Get the observed diensity profile

    rlowers=[]
    rmids=[]
    ruppers=[]
    obs=[]
    params=[]
    ndims=[]
    sigmas=[]
    kernels=[]

    for oparam in observations:

        rlower,rmid,rupper,ob,param,ndim,sigma,obkernel=observations[oparam]
        rlowers.append(rlower)
        rmids.append(rmid)
        ruppers.append(rupper)
        obs.append(ob)
        params.append(param)
        ndims.append(ndim)
        sigmas.append(sigma)
        kernels.append(obkernel)

    mods=[]

    for i in range(0,len(obs)):
        if params[i]=='rho' or params[i]=='Sigma':
            mod=density(stars,rlowers[i],rmids[i],ruppers[i],params[i],ndims[i],kernel=kernels[i],**kwargs)
        elif ('rho' in params[i] or 'Sigma' in params[i]) and ('v' in params[i]):
            mod=density_weighted_mean_squared_velocity(stars,rlowers[i],rmids[i],ruppers[i],params[i],ndims[i],kernel=kernels[i],**kwargs)
        else:
            mod=mean_squared_velocity(stars,rlowers[i],rmids[i],ruppers[i],params[i],ndims[i],kernel=kernels[i],**kwargs)
        mods.append(mod)

    #Entropy:Bovy Equation 21
    #S=-mu*np.sum(stars.mass.value_in(units.MSun)*np.log(stars.mass.value_in(units.MSun)/w0.value_in(units.MSun)-1.))
    dsdw=-mu*np.log(stars.mass.value_in(units.MSun)/w0)

    #Calculate delta_j,velocities (if necessary) and radii for each observable
    delta_j=[]
    v2s=[]
    rs=[]

    for i in range(0,len(obs)):
        delta_j.append(mods[i]-obs[i])

        if params[i]=='rho' or params[i]=='Sigma':
            v2s.append([None]*len(stars))
        else:
            v2s.append(get_v2(stars,params[i],ndims[i]))

        if ndims[i]==3:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
        elif ndims[i]==2:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)
        elif ndims[i]==1:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)

        rs.append(r)


    #Bovy Equation 22/23
    if delta_j_tilde is None or alpha==0.:
        delta_j_tilde=[]
        for i in range(0,len(delta_j)):
            delta_j_tilde.append(delta_j[i])
    else:
        for i in range(0,len(delta_j_tilde)):
            delta_j_tilde[i]+=step*alpha*(delta_j[i]-delta_j_tilde[i])


    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Find dwdt for each star
    for i in range(0,len(stars)):

        K_j=[]

        for j in range(0,len(obs)):
            K_j.append(get_kernel(rs[j][i],rlowers[j],rmids[j],ruppers[j],kernel=kernels[j],ndim=ndims[j],**kwargs))

        dchi2=[]

        for j in range(0,len(obs)):
            dchi2.append(get_dchi2(delta_j_tilde[j],K_j[j],sigmas[j],v2s[j][i]))

        #Bovy Equation 18
        dchisum=0

        for j in range(0,len(dchi2)):
            dchisum+=np.sum(dchi2[j])/2.

        if debug and i==0: print(i,delta_j_tilde,K_j,sigmas,v2s[j][i])

        dwdt[i]=epsilon*stars[i].mass.value_in(units.MSun)*(dsdw[i]-dchisum)


        stars[i].mass += step*dwdt[i] | units.MSun

    #Sum over Chi2 contributions
    chi_squared=0.
    for i in range(0,len(obs)):
        chi_squared+=np.sum((delta_j_tilde[i]/sigmas[i])**2.)

    return stars,chi_squared,delta_j_tilde

def get_dchi2(delta_j_tilde,K_j,sigma_j,v2=None):

    #From Bovy Equations 19,20, and A3
    if v2 is not None:
        dchi2=2.0*delta_j_tilde*v2*K_j/(sigma_j**2.)
    else:
        dchi2=2.0*delta_j_tilde*K_j/(sigma_j**2.)

    return dchi2








