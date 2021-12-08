import numpy as np
from amuse.lab import *
from amuse.units import nbody_system,units
from amuse.ext.LagrangianRadii import LagrangianRadii
import operator

import galpy.util.bovy_coords as coords

from .kernels import *


def get_dynamical_time_scale(Mcluster, Rcluster, G=constants.G):
    return np.sqrt(Rcluster**3/(G*Mcluster))

def density(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix',**kwargs):

    if kernel=='standard':

        return standard_density(particles,rlower=rlower,rmid=rmid,rupper=rupper,param=param,ndim=ndim,nbin=nbin,bins=bins, bintype=bintype)

    else:

        if ndim==3:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
        else:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

        if rlower is None:

            if bintype=='num':
                rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
            elif bintype =='fix':
                rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)
        else:
            nbin=len(rlower)

        rho=np.zeros(nbin)

        for i in range(0,len(r)):
            K_j=get_kernel(r[i],rlower,rmid,rupper,kernel,ndim,**kwargs)
            rho+=particles[i].mass.value_in(units.MSun)*K_j

        if bins:
            return rlower,rmid,rupper,rho
        else:
            return rho

def standard_density(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,bins=False, bintype='fix'):

    if ndim==3:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
    else:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)


    if rlower is None:

        if bintype=='num':
            rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
        elif bintype =='fix':
            rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)

    rho=np.array([])

    for i in range(0,len(rmid)):
                    
        indx=(r > rlower[i]) * (r <= rupper[i])
        msum=np.sum(particles[indx].mass).value_in(units.MSun)

        if ndim==3:
            vol=4.0*np.pi*(rupper[i]**3-rlower[i]**3.)/3.
            dens=msum/vol

        elif ndim==2:
            area=np.pi*(rupper[i]**2.)-np.pi*(rlower[i]**2.)
            dens=msum/area

        rho=np.append(rho,dens)

    if bins:
        return rlower,rmid,rupper,rho
    else:
        return rho

def mean_velocity(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix',**kwargs):

    if kernel != 'standard':

        if ndim==3:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
        else:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

        v=get_v(particles,param,ndim)


        if rlower is None:

            if bintype=='num':
                rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
            elif bintype =='fix':
                rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)
        else:
            nbin=len(rlower)

        vprof=np.zeros(nbin)

        #Normalizing kernel
        vnorm=np.zeros(nbin)
        rlowern=kwargs.get('rlowern',rlower)
        rmidn=kwargs.get('rmidn',rmid)
        ruppern=kwargs.get('ruppern',rupper)
        kerneln=kwargs.get('kerneln',kernel)
        ndimn=kwargs.get('ndimn',ndim)

        for i in range(0,len(r)):
            K_j=get_kernel(r[i],rlower,rmid,rupper,kernel,ndim,**kwargs)
            vprof+=particles[i].mass.value_in(units.MSun)*v[i]*K_j


            K_jn=get_kernel(r[i],rlowern,rmidn,ruppern,kerneln,ndimn,**kwargs)
            vnorm+=particles[i].mass.value_in(units.MSun)*K_jn

        divindx=vnorm>0

        vprof[divindx]/=vnorm[divindx]

        if bins:
            return rlower,rmid,rupper,vprof
        else:
            return vprof

    elif kernel=='standard':

        return standard_mean_velocity(particles,rlower=rlower,rmid=rmid,rupper=rupper,param=param,ndim=ndim,nbin=nbin,bins=bins, bintype=bintype)


def standard_mean_velocity(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,bins=False,bintype='fix'):

    if ndim==3:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
    else:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

    v=get_v(particles,param,ndim)

    if rlower is None:

        if bintype=='num':
            rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
        elif bintype =='fix':
            rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)
    else:
        nbin=len(rlower)

    vprof=np.array([])

    for i in range(0,len(rmid)):
                    
        indx=(r > rlower[i]) * (r <= rupper[i])
        
        if np.sum(indx)>0:
            vprof=np.append(vprof,np.mean(v[indx]))
        else:
            vprof=np.append(vprof,0.0)

    if bins:
        return rlower,rmid,rupper,vprof
    else:
        return vprof


def weighted_mean_relative_velocity(particles=None,rlower=None,rmid=None,rupper=None,vprof=None,param=None,ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix',**kwargs):

    if vprof is None:
        if bins:
            vprof=mean_velocity(particles,rlower=rlower,rmid=rmid,rupper=rupper,param=param,ndim=ndim,nbin=nbin,kernel=kernel,bins=bins, bintype=bintype,**kwargs)
        else:
            rlower,rmid,rupper,vprof=mean_velocity(particles,rlower=rlower,rmid=rmid,rupper=rupper,param=param,ndim=ndim,nbin=nbin,kernel=kernel,bins=bins, bintype=bintype,**kwargs)

    if ndim==3:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
    else:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

    v=get_v(particles,param,ndim)

    dvj=np.zeros(len(vprof))

    for i in range(0,len(v)):
        K_j=get_kernel(r[i],rlower,rmid,rupper,kernel,ndim,**kwargs)
        dvj+=(v[i]-vprof)*particles[i].mass.value_in(units.MSun)*K_j

    return dvj

def mean_squared_velocity(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix',rlower_rho=None,rmid_rho=None,rupper_rho=None,kernel_rho=None,ndim_rho=None,norm=False,**kwargs):

    if kernel != 'standard':

        if ndim==3:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
        else:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

        v2=get_v2(particles,param,ndim)


        if rlower is None:

            if bintype=='num':
                rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
            elif bintype =='fix':
                rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)
        else:
            nbin=len(rlower)

        if rlower_rho is None:
            rlower_rho=rlower
            rmid_rho=rmid
            rupper_rho=rupper
            kernel_rho=kernel
            ndim_rho=ndim

        rhov2prof=density_weighted_mean_squared_velocity(particles,rlower=rlower_rho,rmid=rmid_rho,rupper=rupper_rho,param='rho%s' % param,ndim=ndim_rho,nbin=len(rmid_rho),kernel=kernel_rho,bins=False, bintype=bintype, **kwargs) 

        #Normalizing denominator

        v2norm=np.zeros(nbin)

        for i in range(0,len(r)):
            K_j=get_kernel(r[i],rlower,rmid,rupper,kernel,ndim,**kwargs)
            v2norm+=particles[i].mass.value_in(units.MSun)*K_j
        divindx=(v2norm>0)

        v2prof=np.zeros(nbin)

        if np.sum(divindx) == 0:
            print('DIVIDE BY ZERO ERROR',divindx,v2norm)
            return -1
        else:
            v2prof[divindx]=np.asarray(rhov2prof)[divindx]/v2norm[divindx]

        if bins:
            if norm:
                return rlower,rmid,rupper,v2prof,v2norm
            else:
                return rlower,rmid,rupper,v2prof
        else:
            if norm:
                return v2prof,v2norm
            else:
                return v2prof

    elif kernel=='standard':
        return standard_mean_squared_velocity(particles,rlower=rlower,rmid=rmid,rupper=rupper,param=param,ndim=ndim,nbin=nbin,bins=bins, bintype=bintype)


def standard_mean_squared_velocity(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,bins=False,bintype='fix'):

    if ndim==3:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
    else:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

    v2=get_v2(particles,param,ndim)

    if rlower is None:

        if bintype=='num':
            rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
        elif bintype =='fix':
            rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)
    else:
        nbin=len(rlower)

    v2prof=np.array([])

    for i in range(0,len(rmid)):
                    
        indx=(r > rlower[i]) * (r <= rupper[i])
        
        if np.sum(indx)>0:
            v2prof=np.append(v2prof,np.mean(v2[indx]))
        else:
            v2prof=np.append(v2prof,0.0)

    if bins:
        return rlower,rmid,rupper,v2prof
    else:
        return v2prof


def density_weighted_mean_squared_velocity(particles,rlower=None,rmid=None,rupper=None,param='rhov2',ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix', **kwargs):

    if kernel!='standard':

        if ndim==3:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
        else:
            r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

        v2=get_v2(particles,param,ndim)

        if rlower is None:

            if bintype=='num':
                rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
            elif bintype =='fix':
                rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)
        else:
            nbin=len(rlower)

        rhov2prof=np.zeros(nbin)

        for i in range(0,len(r)):
            K_j=get_kernel(r[i],rlower,rmid,rupper,kernel,ndim,**kwargs)
            rhov2prof+=particles[i].mass.value_in(units.MSun)*v2[i]*K_j

        if bins:
            return rlower,rmid,rupper,rhov2prof
        else:
            return rhov2prof

    else:

        return standard_density_weighted_mean_squared_velocity(particles,rlower=rlower,rmid=rmid,rupper=rupper,param=param,ndim=ndim,nbin=nbin,kernel=kernel,bins=bins, bintype=bintype)

def standard_density_weighted_mean_squared_velocity(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,bins=False,bintype='fix'):

    if ndim==3:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)
    else:
        r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.)

    v2=get_v2(particles,param,ndim)

    if rlower is None:

        if bintype=='num':
            rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
        elif bintype =='fix':
            rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)
    else:
        nbin=len(rlower)

    rhoprof=standard_density(particles,rlower=rlower,rmid=rmid,rupper=rupper,param=param,ndim=ndim,nbin=nbin,bins=False, bintype=bintype)

    v2prof=np.array([])

    for i in range(0,len(rmid)):
                    
        indx=(r > rlower[i]) * (r <= rupper[i])
        
        if np.sum(indx)>0:
            v2prof=np.append(v2prof,np.mean(v2[indx]))
        else:
            v2prof=np.append(v2prof,0.0)

    v2prof*=rhoprof

    if bins:
        return rlower,rmid,rupper,v2prof
    else:
        return v2prof

def get_v2(particles,param,ndim):

    if param is None:
        if ndim==3:
            v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.+(particles.vz.value_in(units.kms))**2.)
        elif ndim==2:
            v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.)
        elif ndim==1:
            v=particles.vx.value_in(units.kms)
    else:

        if 'vlos' in param or 'vz' in param:
            v=particles.vz.value_in(units.kms)

        elif 'vR' in param or 'vT' in param:

            R,theta,z=coords.rect_to_cyl(particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec))

            vR,vT,vz=coords.rect_to_cyl_vec(particles.vx.value_in(units.kms),particles.vy.value_in(units.kms),particles.vz.value_in(units.kms),particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec))

            if 'vR' in param:
                v=vR
            elif 'vT' in param:
                v=vT

        elif 'v' in param:
            if ndim==3:
                v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.+(particles.vz.value_in(units.kms))**2.)
            elif ndim==2:
                v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.)
            elif ndim==1:
                v=particles.vx.value_in(units.kms)

    return v*v

def get_v(particles,param,ndim):

    if param is None:
        if ndim==3:
            v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.+(particles.vz.value_in(units.kms))**2.)
        elif ndim==2:
            v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.)
        elif ndim==1:
            v=particles.vx.value_in(units.kms)
    else:

        if 'vlos' in param or 'vz' in param:
            v=particles.vz.value_in(units.kms)

        elif 'vR' in param or 'vT' in param:

            R,theta,z=coords.rect_to_cyl(particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec))

            vR,vT,vz=coords.rect_to_cyl_vec(particles.vx.value_in(units.kms),particles.vy.value_in(units.kms),particles.vz.value_in(units.kms),particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec))

            if 'vR' in param:
                v=vR
            elif param=='vT':
                v=vT

        elif 'v' in param:
            if ndim==3:
                v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.+(particles.vz.value_in(units.kms))**2.)
            elif ndim==2:
                v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.)
            elif ndim==1:
                v=particles.vx.value_in(units.kms)

    return v


def nbinmaker(x, nbin=10, nsum=False):
    """Split an array into bins with equal numbers of elements

    Parameters
    ----------
    x : float
      input array
    nbin : int
      number of bins
    nsum : bool
      return number of points in each bin (default: False)

    Returns
    -------
    x_lower : float
      lower bin values
    x_mid : float
      mean value in each bin
    x_upper : float
      upper bin values
    x_hist : 
      number of points in bin

    if nsum==True:
      x_sum : float
        sum of point values in each bin

    History
    -------
    2018 - Written - Webb (UofT)
    """
    x = np.asarray(x)

    xorder = np.argsort(x)

    x_lower = np.array([])
    x_upper = np.array([])
    x_hist = np.array([])
    x_sum = np.array([])
    x_mid = np.array([])

    for i in range(0, nbin):
        indx = int(float(i) * float(len(x)) / float(nbin))
        x_lower = np.append(x_lower, x[xorder[indx]])

    x_upper=x_lower[1:]
    x_upper=np.append(x_upper,np.amax(x))

    indx = x_lower != x_upper
    x_lower = x_lower[indx]
    x_upper = x_upper[indx]

    for i in range(0, np.sum(indx)):
        if i<np.sum(indx)-1:
            xindx = (x >= x_lower[i]) * (x < x_upper[i])
        else:
            xindx = (x >= x_lower[i])

        x_hist = np.append(x_hist, np.sum(xindx))
        x_sum = np.append(x_sum, np.sum(x[xindx]))
        x_mid = np.append(x_mid, x_sum[i] / x_hist[i])

    if nsum:
        return x_lower, x_mid, x_upper, x_hist, x_sum
    else:
        return x_lower, x_mid, x_upper, x_hist


def binmaker(x, nbin=10, nsum=False, steptype="linear"):
    """Split an array into bins of equal width

    Parameters
    ----------
    x : float
      input array
    nbin : int
      number of bins
    nsum : bool
      return number of points in each bin (default: False)
    steptype : str
      linear or logarithmic steps (default: linear)

    Returns
    -------
    x_lower : float
      lower bin values
    x_mid : float
      mean value in each bin
    x_upper : float
      upper bin values
    x_hist : 
      number of points in bin

    if nsum==True:
      x_sum : float
        sum of point values in each bin

    History
    -------
    2018 - Written - Webb (UofT)
    """

    x_hist = np.zeros(nbin)
    x_sum = np.zeros(nbin)
    x = np.array(x)

    if steptype == "linear":
        steps = np.linspace(np.amin(x), np.amax(x), nbin + 1)
    else:
        steps = np.logspace(np.log10(np.amin(x)), np.log10(np.amax(x)), nbin + 1)

    x_lower = steps[:-1]
    x_upper = steps[1:]

    x_mid = (x_upper + x_lower) / 2.0

    for j in range(0, nbin):
        if j<nbin-1:
            indx = (x >= x_lower[j]) * (x < x_upper[j])
        else:
            indx = (x >= x_lower[j]) * (x <= x_upper[j])

        x_hist[j] = len(x[indx])
        x_sum[j] = np.sum(x[indx])

    if nsum:
        return x_lower, x_mid, x_upper, x_hist, x_sum
    else:
        return x_lower, x_mid, x_upper, x_hist

def chi2(obs,mod):
    return np.sum((mod-obs)2./obs)


