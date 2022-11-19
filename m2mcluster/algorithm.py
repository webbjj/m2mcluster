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

def made_to_measure(stars,observations,models,norms,w0,epsilon=10.0**-4.,mu=1.,alpha=1.,step=1.,delta_j_tilde=None,debug=False,**kwargs,):
    
    ndebug=kwargs.get('ndebug',1)

    #Initialize rate of change in weights within each radial bin to be zero
    dwdt=np.zeros(len(stars))

    #Entropy:Bovy Equation 21
    #S=-mu*np.sum(stars.mass.value_in(units.MSun)*np.log(stars.mass.value_in(units.MSun)/w0.value_in(units.MSun)-1.))
    dsdw=-mu*np.log(stars.mass.value_in(units.MSun)/w0)
    dwdt=epsilon*stars.mass.value_in(units.MSun)*dsdw

    #Sum over Chi2 contributions
    chi_squared=0.



    for j,oparam in enumerate(observations):

        rlower,rmid,rupper,obs,param,ndim,sigma,kernel=observations[oparam]

        mod=models[oparam]
        norm=norms[oparam]

    #Calculate delta_j,velocities (if necessary) and radii for each observable
        print('LEN MOD, LEN OBS, PARAM')
        print(len(mod),len(obs),param)
        delta_j=(mod-obs)

        if ('rho' in param or 'Sigma' in param) and ('v' not in param):
            v2=None
        else:
            v2=(get_v2(stars,param,ndim,))

        if ndim==3:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.+(stars.z.value_in(units.parsec))**2.)
        elif ndim==2:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)
        elif ndim==1:
            r=np.sqrt((stars.x.value_in(units.parsec))**2.+(stars.y.value_in(units.parsec))**2.)



        #Bovy Equation 22/23
        if debug:
            print('ALPHA,DELTA_J_TILDE,DELTA_J,STEP')
            print(alpha,delta_j_tilde[j],delta_j,step)
        if alpha==0.:
            delta_j_tilde[j]=delta_j
        else:
            delta_j_tilde[j]+=step*alpha*(delta_j-delta_j_tilde[j])

        chi_squared+=np.sum((delta_j_tilde[j]/sigma)**2.)

        #Find dwdt for each star

        K_j=(get_kernel(r,rlower,rmid,rupper,kernel=kernel,ndim=ndim,**kwargs))
        if ('rho' in param or 'Sigma' in param) and ('v' not in param):
            dv=None

        elif 'v' in param and '2' in param and 'rho' not in param and 'Sigma' not in param:
            dv=v2.reshape(len(v2),1)-mod
            dv=np.swapaxes(dv,0,1)
        else:
            dv=None

        delta_j_tildes=np.array(delta_j_tilde[j]).reshape(len(delta_j_tilde[j]),1)
        sigmas=np.array(sigma).reshape(len(sigma),1)

        dchi2=get_dchi2(delta_j_tildes,K_j,sigmas,v2,dv)

        if norm is not None:
            divindx=norm!=0
            normrs=norm.reshape(len(norm),1)
            dchi2[divindx]/=normrs[divindx]

            if np.sum(divindx) != len(divindx):
                dchi2[np.invert(divindx)]=0.


        #Bovy Equation 18
   
        dchisum=np.sum(dchi2,axis=0)/2.

        if debug: 
            print('NDEBUG,R,DELTA_J_TILDE,K_J,SIGMA,v2,dv,DCHISUM')
            if dv is None:
                if v2 is None:
                    print(ndebug,r[ndebug],delta_j_tilde[j],K_j[:,ndebug],sigma,None,None,dchisum[ndebug])
                else:
                    print(ndebug,r[ndebug],delta_j_tilde[j],K_j[:,ndebug],sigma,v2[ndebug],None,dchisum[ndebug])
            else:
                print(ndebug,r[ndebug],delta_j_tilde[j],K_j[:,ndebug],sigma,v2[ndebug],dv[:,ndebug],dchisum[ndebug])

        dwdt-=epsilon*stars.mass.value_in(units.MSun)*dchisum

        if debug: print('DWDT: ',-epsilon*stars[ndebug].mass.value_in(units.MSun)*dchisum[ndebug],dwdt[ndebug])

    stars.mass += step*dwdt | units.MSun


    return stars,chi_squared,delta_j_tilde,dwdt

def get_dchi2(delta_j_tilde,K_j,sigma_j,v2=None,dv=None):
    
    if dv is not None:
    #From Bovy Equations  A3
        dchi2=2.0*delta_j_tilde*dv*K_j/(sigma_j**2.)
    
    elif v2 is not None:
    #From Bovy Equations 19,20

        dchi2=2.0*delta_j_tilde*v2*K_j/(sigma_j**2.)
    else:
        dchi2=2.0*delta_j_tilde*K_j/(sigma_j**2.)

    return dchi2








