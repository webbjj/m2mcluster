import numpy as np
from amuse.lab import *
from amuse.units import nbody_system,units
from amuse.ext.LagrangianRadii import LagrangianRadii
import operator

def get_dynamical_time_scale(Mcluster, Rcluster, G=constants.G):
    return np.sqrt(Rcluster**3/(G*Mcluster))

def density(particles,rlower=None,rmid=None,rupper=None,ndim=2,nbin=20,bins=False, bintype='fix'):

    r=np.sqrt((particles.x.value_in(units.parsec))**2.+(particles.y.value_in(units.parsec))**2.+(particles.z.value_in(units.parsec))**2.)

    if rlower is None:

        print('DEBUG:',len(r),len(particles),np.sum(particles.mass.value_in(units.MSun) > 0))

        if bintype=='num':
            rlower, rmid, rupper, rhist=nbinmaker(r,nbin=nbin)
        elif bintype =='fix':
            rlower, rmid, rupper, rhist=binmaker(r,nbin=nbin)

    rho=np.array([])

    for i in range(0,len(rmid)):
                    
        indx=(r > rlower[i]) * (r <= rupper[i])

        if ndim==2:
            area=np.pi*(rupper[i]**2.)-np.pi*(rlower[i]**2.)
        elif ndim==3:
            area=4.0*np.pi*(rupper[i]**3-rlower[i]**3.)/3.

        msum=np.sum(particles[indx].mass).value_in(units.MSun)
        dens=msum/area
        rho=np.append(rho,dens)

    if bins:
        return rlower,rmid,rupper,rho
    else:
        return rho

def chi2(mod,obs,sigma=None):

    delta_j=mod/obs-1.

    if sigma is None:
        chi_squared=np.sum((delta_j)**2.)/len(delta_j)
    else:
        chi_squared=np.sum((delta_j/sigma)**2.)/len(delta_j)

    return chi_squared


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