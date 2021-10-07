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

def made_to_measure(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,mscale=1.,zeta=None,xi=None,step=1.,delta_j_tilde=None,method='Seyer',debug=False,**kwargs,):
    
    #JO - alpha >= epsilon?
    #JO - alpha=1/dt is equivalent to no smoothing and alpha cannot be set to a larger value. Which dt? integartion time or eval time?

    if method=='Seyer':

        stars,chi_squared,delta_j_tilde=made_to_measure_seyer(stars,observations,w0,epsilon,mu,alpha,step,delta_j_tilde,debug,**kwargs)
        return stars,chi_squared,delta_j_tilde

    elif method=='Hunt':
        stars,chi_squared,delta_j_tilde=made_to_measure_hunt(stars,observations,w0,epsilon,mu,alpha,mscale,zeta,xi,step,delta_j_tilde,debug,**kwargs)
        return stars,chi_squared,delta_j_tilde
    else:

        stars,chi_squared,delta_j_tilde=made_to_measure_bovy(stars,observations,w0,epsilon,mu,alpha,step,delta_j_tilde,debug,**kwargs)
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
    if alpha==0.:
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

        if param=='rho' or params=='Sigma':
            rlower_rho=rlower
            rmid_rho=rmid
            rupper_rho=rupper
            kernel_rho=obkernel
            ndim_rho=ndim

    mods=[]
    norms=[]

    for j in range(0,len(obs)):
        if params[j]=='rho' or params[j]=='Sigma':
            mod=density(stars,rlowers[j],rmids[j],ruppers[j],params[j],ndims[j],kernel=kernels[j],**kwargs)
            norms.append([None]*len(rmids[j]))
        elif ('rho' in params[j] or 'Sigma' in params[j]) and ('v' in params[j]) and ('2' in params[j]):
            mod=density_weighted_mean_squared_velocity(stars,rlowers[j],rmids[j],ruppers[j],params[j],ndims[j],kernel=kernels[j],**kwargs)
            norms.append([None]*len(rmids[j]))
        elif ('v' in params[j]) and ('2' in params[j]):
            mod,norm=mean_squared_velocity(stars,rlowers[j],rmids[j],ruppers[j],params[j],ndims[j],kernel=kernels[j],rlower_rho=rlower_rho,rmid_rho=rmid_rho,rupper_rho=rupper_rho,kernel_rho=kernel_rho,ndim_rho=ndim_rho,norm=True,**kwargs)
            norms.append(norm)
        mods.append(mod)

    #Entropy:Bovy Equation 21
    #S=-mu*np.sum(stars.mass.value_in(units.MSun)*np.log(stars.mass.value_in(units.MSun)/w0.value_in(units.MSun)-1.))
    dsdw=-mu*np.log(stars.mass.value_in(units.MSun)/w0)

    #Calculate delta_j,velocities (if necessary) and radii for each observable
    delta_j=[]
    v2s=[]
    rs=[]

    for j in range(0,len(obs)):
        delta_j.append(mods[j]-obs[j])

        if params[j]=='rho' or params[j]=='Sigma':
            v2s.append([None]*len(stars))
        else:
            v2s.append(get_v2(stars,params[j],ndims[j]))

        if ndims[j]==3:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
        elif ndims[j]==2:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)
        elif ndims[j]==1:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)

        rs.append(r)


    #Bovy Equation 22/23
    if alpha==0.:
        for j in range(0,len(delta_j_tilde)):
            delta_j_tilde[j]=delta_j[j]
    else:
        for j in range(0,len(delta_j_tilde)):
            delta_j_tilde[j]+=step*alpha*(delta_j[j]-delta_j_tilde[j])


    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Find dwdt for each star
    for i in range(0,len(stars)):

        K_j=[]
        dv=[]

        for j in range(0,len(obs)):
            K_j.append(get_kernel(rs[j][i],rlowers[j],rmids[j],ruppers[j],kernel=kernels[j],ndim=ndims[j],**kwargs))
            if params[j]=='rho' or params[j]=='Sigma':
                K_j_rho=K_j[-1]
                dv.append([None]*len(mods[j]))

            elif 'v' in params[j] and '2' in params[j] and 'rho' not in params[j]:
                dv.append(v2s[j][i]-mods[j])
            else:
                dv.append([None]*len(mods[j]))

        dchi2=[]

        for j in range(0,len(obs)):

            dchi2.append(get_dchi2(delta_j_tilde[j],K_j[j],sigmas[j],v2s[j][i],dv[j]))

            if not np.array_equal(norms[j],[None]*len(norms[j])):
                divindx=norms[j]!=0
                if debug and i==0: print('NORMALIZING MEAN SQUARE VELOCITY',norms[j])
                dchi2[-1][divindx]/=norms[j][divindx]

                if np.sum(divindx) != len(divindx):
                    dchi2[-1][np.invert(divindx)]=0.


        #Bovy Equation 18
        dchisum=0

        for j in range(0,len(dchi2)):
            dchisum+=np.sum(dchi2[j])/2.

        if debug and i==0: print(i,rs[:][i],delta_j_tilde,K_j,sigmas,v2s[:][i])

        dwdt[i]=epsilon*stars[i].mass.value_in(units.MSun)*(dsdw[i]-dchisum)


        stars[i].mass += step*dwdt[i] | units.MSun

    #Sum over Chi2 contributions
    chi_squared=0.
    for j in range(0,len(obs)):
        chi_squared+=np.sum((delta_j_tilde[j]/sigmas[j])**2.)

    return stars,chi_squared,delta_j_tilde

def made_to_measure_hunt(stars,observations,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,mscale=1.,zeta=1.,xi=None,step=1.,delta_j_tilde=None,debug=False,**kwargs,):
    
    #Make sure zeta and xi are set:
    if xi is None:
        print('PLEASE SET ZETA AND ALL VALUES OF XI')
        return -1

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

        if oparam=='rho' or oparam=='sigma':
            obs_dens=ob


    mods=[]

    for j in range(0,len(obs)):
        if params[j]=='rho' or params[j]=='Sigma':
            mod=density(stars,rlowers[j],rmids[j],ruppers[j],params[j],ndims[j],kernel=kernels[j],**kwargs)
        elif ('v' in params[j]):
            mod=weighted_mean_relative_velocity(particles=stars,rlower=rlowers[j],rmid=rmids[j],rupper=ruppers[j],vprof=obs[j],param=params[j],ndim=ndims[j],kernel=kernels[j],bins=False,**kwargs)

        mods.append(mod)

    #S=-mu*np.sum(stars.mass.value_in(units.MSun)*np.log(stars.mass.value_in(units.MSun)/w0.value_in(units.MSun)-1.))
    dsdw=-mu*(np.log(stars.mass.value_in(units.MSun)/w0)+1.)

    #Calculate delta_j,velocities (if necessary) and radii for each observable
    delta_j=[]
    vs=[]
    rs=[]

    nv=0
    dvobs=kwargs.get('dvobs')

    for j in range(0,len(obs)):

        if params[j]=='rho' or params[j]=='Sigma':
            delta_j.append(mods[j]/obs[j]-1.)
            vs.append([None]*len(stars))
        elif 'v' in params[j]:
            delta_j.append((mods[j]-dvobs[nv])/(sigmas[j]*obs_dens))
            vs.append(get_v(stars,params[j],ndims[j]))
            nv+=1

        if ndims[j]==3:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
        elif ndims[j]==2:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)
        elif ndims[j]==1:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)

        rs.append(r)

    #Bovy Equation 22/23
    if delta_j_tilde is None or alpha==0.:
        for j in range(0,len(delta_j_tilde)):
            delta_j_tilde[j]=delta_j[j]
    else:
        for j in range(0,len(delta_j_tilde)):
            delta_j_tilde[j]+=step*alpha*(delta_j[j]-delta_j_tilde[j])


    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Find dwdt for each star
    for i in range(0,len(stars)):

        K_j=[]

        for j in range(0,len(obs)):
            K_j.append(get_kernel(rs[j][i],rlowers[j],rmids[j],ruppers[j],kernel=kernels[j],ndim=ndims[j],**kwargs))

        dc_v=[]
        nv=0

        for j in range(0,len(obs)):
            if params[j]=='rho' or params[j]=='Sigma':
                if debug and i==0: print('RHOTEST: ',K_j[j],delta_j_tilde[j],obs[j])

                dc_rho=mscale*K_j[j]*delta_j_tilde[j]/obs[j]
                if debug and i==0: print('dc_rho',dc_rho)

            else:
                delta=vs[j][i]-obs[j]
                if debug and i==0: print('VTEST: ', K_j[j],delta_j_tilde[j],delta,sigmas[j],obs_dens)
                dc_v.append(xi[nv]*K_j[j]*delta_j_tilde[j]*delta/(sigmas[j]*obs_dens))
                if debug and i==0: print('dc_v',nv,dc_v[-1])
                nv+=1

        #Bovy Equation 18
        dchisum=np.sum(dc_rho)
        for j in range(0,nv):
            dchisum+=mscale*zeta*np.sum(dc_v[j])

        if debug and i==0: 
            for j in range(0,len(obs)):
                print('FULL ',i,j,delta_j_tilde[j],K_j[j],sigmas[j],vs[j][i])

        dwdt[i]=epsilon*stars[i].mass.value_in(units.MSun)*(dsdw[i]-dchisum)

        if debug and i==0: 
            print('DWDT: ',dwdt[i],dsdw[i],np.sum(dc_rho))
            for j in range(0,nv):
                print(mscale*zeta*np.sum(dc_v[j]))



        stars[i].mass += step*dwdt[i] | units.MSun

    #Sum over Chi2 contributions
    chi_squared=0.
    for j in range(0,len(obs)):
        chi_squared+=np.sum((delta_j_tilde[j]/sigmas[j])**2.)

    return stars,chi_squared,delta_j_tilde

def get_dchi2(delta_j_tilde,K_j,sigma_j,v2=None,dv=None):

    if not np.array_equal(dv,[None]*len(dv)):
    #From Bovy Equations  A3
        dchi2=2.0*delta_j_tilde*dv*K_j/(sigma_j**2.)
    
    elif v2 is not None:
    #From Bovy Equations 19,20

        dchi2=2.0*delta_j_tilde*v2*K_j/(sigma_j**2.)
    else:
        dchi2=2.0*delta_j_tilde*K_j/(sigma_j**2.)

    return dchi2








