import numpy as np
from amuse.lab import *
from amuse.units import nbody_system,units
from amuse.ext.LagrangianRadii import LagrangianRadii
import operator

import galpy.util.bovy_coords as coords

from .kernels import *

from clustertools import cart_to_sphere,cart_to_cyl

def density(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix',**kwargs):

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

    K_j=get_kernel(r,rlower,rmid,rupper,kernel,ndim,**kwargs)
    rho=np.sum(particles.mass.value_in(units.MSun)*K_j,axis=1)

    if bins:
        return rlower,rmid,rupper,rho
    else:
        return rho

def mean_velocity(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix',**kwargs):


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

    K_j=get_kernel(r,rlower,rmid,rupper,kernel,ndim,**kwargs)
    vprof=np.sum(particles.mass.value_in(units.MSun)*v*K_j,axis=1)


    K_jn=get_kernel(r,rlowern,rmidn,ruppern,kerneln,ndimn,**kwargs)
    vnorm=np.sum(particles.mass.value_in(units.MSun)*K_jn,axis=1)

    divindx=vnorm>0

    vprof[divindx]/=vnorm[divindx]

    if bins:
        return rlower,rmid,rupper,vprof
    else:
        return vprof

def mean_squared_velocity(particles,rlower=None,rmid=None,rupper=None,param=None,ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix',norm=False,**kwargs):

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

    rhov2prof=density_weighted_mean_squared_velocity(particles,rlower=rlower,rmid=rmid,rupper=rupper,param='rho%s' % param,ndim=ndim,nbin=len(rmid),kernel=kernel,bins=False, bintype=bintype, **kwargs) 

    #Normalizing denominator

    v2norm=np.zeros(nbin)


    K_j=get_kernel(r,rlower,rmid,rupper,kernel,ndim,**kwargs)
    v2norm=np.sum(particles.mass.value_in(units.MSun)*K_j,axis=1)

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

def density_weighted_mean_squared_velocity(particles,rlower=None,rmid=None,rupper=None,param='rhov2',ndim=3,nbin=20,kernel='identifier',bins=False, bintype='fix', **kwargs):

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

    K_j=get_kernel(r,rlower,rmid,rupper,kernel,ndim,**kwargs)
    rhov2prof=np.sum(particles.mass.value_in(units.MSun)*v2*K_j,axis=1)

    if bins:
        return rlower,rmid,rupper,rhov2prof
    else:
        return rhov2prof

def get_v2(particles,param,ndim):

    if param is None:
        if ndim==3:
            v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.+(particles.vz.value_in(units.kms))**2.)
        elif ndim==2:
            v=np.sqrt((particles.vx.value_in(units.kms))**2.+(particles.vy.value_in(units.kms))**2.)
        elif ndim==1:
            v=particles.vx.value_in(units.kms)
    else:

        if 'vlos' in param:
            v=particles.vx.value_in(units.kms)

        elif 'vr' in param or 'vp' in param or 'vt' in param:

            r,phi,theta,vr,vp,vt=cart_to_sphere(particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec),particles.vx.value_in(units.kms),particles.vy.value_in(units.kms),particles.vz.value_in(units.kms))


            if 'vr' in param:
                v=vr
            elif 'vp' in param:
                v=vp
            elif 'vt' in param:
                v=vt

        elif 'vR' in param or 'vT' in param or 'vz' in param:

            R, theta, z, vR, vT, vz=cart_to_cyl(particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec),particles.vx.value_in(units.kms),particles.vy.value_in(units.kms),particles.vz.value_in(units.kms))

            if 'vR' in param:
                v=vR
            elif 'vT' in param:
                v=vT
            elif 'vz' in param:
                v=vz

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

        if 'vlos' in param:
            v=particles.vx.value_in(units.kms)

        elif 'vr' in param or 'vp' in param or 'vt' in param:

            r,phi,theta,vr,vp,vt=cart_to_sphere(particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec),particles.vx.value_in(units.kms),particles.vy.value_in(units.kms),particles.vz.value_in(units.kms))


            if 'vr' in param:
                v=vr
            elif 'vp' in param:
                v=vp
            elif 'vt' in param:
                v=vt

        elif 'vR' in param or 'vT' in param or 'vz' in param:

            R, theta, z, vR, vT, vz=cart_to_cyl(particles.x.value_in(units.parsec),particles.y.value_in(units.parsec),particles.z.value_in(units.parsec),particles.vx.value_in(units.kms),particles.vy.value_in(units.kms),particles.vz.value_in(units.kms)) 

            if 'vR' in param:
                v=vR
            elif 'vT' in param:
                v=vT
            elif 'vz' in param:
                v=vz

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
    return np.sum(((mod-obs)**2.)/obs)