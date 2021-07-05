#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as pyplot
from amuse.plot import scatter
import numpy as np
from amuse.units import nbody_system,units
#from galpy.util import bovy_plot

from .functions import density,velocity_dispersion

#import seaborn as sns
#df = sns.load_dataset('iris')

def positions_plot(stars,filename=None):
        

    pyplot.scatter(stars.x.value_in(units.parsec),stars.y.value_in(units.parsec),alpha=0.1)
    pyplot.xlabel('X (pc)')
    pyplot.ylabel('Y (pc)')
    
    #pyplot.xlim(-0.05,0.05)
    #pyplot.ylim(-0.05,0.05)

    if filename is not None:
        pyplot.savefig(filename)
        pyplot.close()
    else:
        pyplot.show()
        pyplot.close()

def density_profile(stars,observed_rho,filename=None):

    rlower,rmid,rupper,rho, param, ndim=observed_rho

    vol=(4./3.)*np.pi*(rupper**3.-rlower**3.)
    area=np.pi*(rupper**2.-rlower**2.)

    mod_rho=density(stars,rlower,rmid, rupper,ndim)

    mod_rlower,mod_rmid,mod_rupper,mod_rho_full=density(stars,ndim=ndim,bins=True,bintype='num')

    #Compare density profiles
    mindx=(mod_rho > 0.)
    pyplot.loglog(rmid[mindx],mod_rho[mindx],'r',label='Model')
    pyplot.loglog(rmid[mindx],mod_rho[mindx],'ro')

    mindx=(mod_rho_full > 0.)
    #pyplot.loglog(mod_rmid[mindx],mod_rho_full[mindx],'r--',label='Model Full')
    #pyplot.loglog(mod_rmid[mindx],mod_rho_full[mindx],'ro')

    mindx=(rho > 0.)

    pyplot.loglog(rmid[mindx],rho[mindx],'k',label='Observations')
    pyplot.loglog(rmid[mindx],rho[mindx],'ko')

    pyplot.legend()
    pyplot.xlabel('$\log_{10} r$ (pc)')

    if ndim==3:
        pyplot.ylabel(r'$\log_{10} \rho$ ($M_{\odot}/pc^3)$')
    else:
        pyplot.ylabel(r'$\log_{10} \Sigma$ ($M_{\odot}/pc^2)$')

    if filename is not None:
        pyplot.savefig(filename)
        pyplot.close()
    else:
        pyplot.show()
        pyplot.close()

def velocity_dispersion_profile(stars,observed_sigv,filename=None):

    rlower,rmid,rupper,sigv, param, ndim=observed_sigv

    vol=(4./3.)*np.pi*(rupper**3.-rlower**3.)
    area=np.pi*(rupper**2.-rlower**2.)

    mod_sigv=velocity_dispersion(stars,rlower,rmid, rupper,ndim)

    mod_rlower,mod_rmid,mod_rupper,mod_sigv_full=velocity_dispersion(stars,ndim=ndim,bins=True,bintype='num')

    #Compare density profiles
    mindx=(mod_sigv > 0.)
    pyplot.loglog(rmid[mindx],mod_sigv[mindx],'r',label='Model')
    pyplot.loglog(rmid[mindx],mod_sigv[mindx],'ro')

    mindx=(mod_sigv_full > 0.)
    #pyplot.loglog(mod_rmid[mindx],mod_sigv_full[mindx],'r--',label='Model Full')
    #pyplot.loglog(mod_rmid[mindx],mod_sigv_full[mindx],'ro')

    mindx=(sigv > 0.)

    pyplot.loglog(rmid[mindx],sigv[mindx],'k',label='Observations')
    pyplot.loglog(rmid[mindx],sigv[mindx],'ko')

    pyplot.legend()
    pyplot.xlabel('$\log_{10} r$ (pc)')
    pyplot.ylabel(r'$\log_{10} \sigma_v$ ($\rm km/s$)')

    if filename is not None:
        pyplot.savefig(filename)
        pyplot.close()
    else:
        pyplot.show()
        pyplot.close()
