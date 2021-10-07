#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as pyplot
from amuse.plot import scatter
import numpy as np
from amuse.units import nbody_system,units
#from galpy.util import bovy_plot

from .functions import density,mean_velocity,mean_squared_velocity,density_weighted_mean_squared_velocity

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

def density_profile(stars,observations,nbin=20,bintype='num',filename=None,**kwargs):

    for oparam in observations:

        if 'rho' == oparam or 'Sigma' == oparam:
            rlower,rmid,rupper,rho, param, ndim, sigma, rhokernel=observations[oparam]

            vol=(4./3.)*np.pi*(rupper**3.-rlower**3.)
            area=np.pi*(rupper**2.-rlower**2.)

            mod_rho=density(stars,rlower,rmid,rupper,param,ndim,kernel=rhokernel,**kwargs)


            #Compare density profiles
            mindx=(mod_rho > 0.) * (rupper < 1.e10)
            pyplot.loglog(rmid[mindx],mod_rho[mindx],'r',label='Model')
            pyplot.loglog(rmid[mindx],mod_rho[mindx],'ro')

            mindx=(rho > 0.) * (rupper < 1.e10)

            pyplot.loglog(rmid[mindx],rho[mindx],'k',label='Observations')
            pyplot.loglog(rmid[mindx],rho[mindx],'ko')

            pyplot.legend()
            pyplot.xlabel('$\log_{10} r$ (pc)')

            if ndim==3:
                pyplot.ylabel(r'$\log_{10} \rho$ ($M_{\odot}/pc^3)$')
            elif ndim==2:
                pyplot.ylabel(r'$\log_{10} \Sigma$ ($M_{\odot}/pc^2)$')

            if filename is not None:
                pyplot.savefig(filename)
                pyplot.close()
            else:
                pyplot.show()
                pyplot.close()

def density_weighted_mean_squared_velocity_profile(stars,observations,nbin=20,bintype='num',filename=None,**kwargs):

    for oparam in observations:
        if (('rhov' in oparam) and ('2' in oparam)) or (('sigmav' in oparam) and ('2' in oparam)):

            rlower,rmid,rupper,v2,param,ndim,sigma, obskernel = observations[oparam]
            mod_v2=density_weighted_mean_squared_velocity(stars,rlower,rmid, rupper, param, ndim, kernel=obskernel,**kwargs)


            #Compare density profiles
            mindx=(mod_v2 > 0.) * (rupper < 1.e10)
            pyplot.loglog(rmid[mindx],mod_v2[mindx],'r',label='Model')
            pyplot.loglog(rmid[mindx],mod_v2[mindx],'ro')

            mindx=(v2 > 0.) * (rupper < 1.e10)

            pyplot.loglog(rmid[mindx],v2[mindx],'k',label='Observations')
            pyplot.loglog(rmid[mindx],v2[mindx],'ko')

            pyplot.legend()
            pyplot.xlabel(r'$\rm \log_{10} r \ (pc)')

            pyplot.ylabel(r'$\rm \log_{10} \rho <v^2> \ (M_{\odot}/pc^3 \ km/s)$')


            if filename is not None:
                pyplot.savefig(filename)
                pyplot.close()
            else:
                pyplot.show()
                pyplot.close()


def mean_squared_velocity_profile(stars,observations,nbin=20,bintype='num',filename=None,**kwargs):

    nv=0

    for oparam in observations:
        if ('v' in oparam) and ('2' in oparam) and ('rhov' not in oparam) and ('sigmav' not in oparam):

            print(oparam)
            rlower,rmid,rupper,v2,param,ndim,sigma, obskernel = observations[oparam]

            mod_v2=mean_squared_velocity(stars,rlower,rmid, rupper, param, ndim, kernel=obskernel,**kwargs)

            #Compare density profiles
            mindx=(mod_v2 > 0.) * (rupper < 1.e10)
            pyplot.loglog(rmid[mindx],mod_v2[mindx],'r',label='Model')
            pyplot.loglog(rmid[mindx],mod_v2[mindx],'ro')

            mindx=(v2 > 0.) * (rupper < 1.e10)

            pyplot.loglog(rmid[mindx],v2[mindx],'k',label='Observations')
            pyplot.loglog(rmid[mindx],v2[mindx],'ko')

            pyplot.legend()
            pyplot.xlabel(r'$\log_{10} r$ (pc)')

            pyplot.ylabel(r'$ \log_{10} <%s^2>$ ($\rm km/s$)' % oparam)

            if filename is not None:

                if nv == 0:
                    pyplot.savefig(filename)
                else:
                    dotspot=filename.find('.')
                    newfilename = filename[:dotspot]+('_%i' % nv)+filename[dotspot:]
                    pyplot.savefig(newfilename)

                pyplot.close()

            else:
                pyplot.show()
                pyplot.close()


            nv+=1

def mean_velocity_profile(stars,observations,nbin=20,bintype='num',filename=None,**kwargs):

    nv=0

    for oparam in observations:
        if ('v' in oparam) and ('2' not in oparam) and ('rhov' not in oparam) and ('sigmav' not in oparam):

            rlower,rmid,rupper,v,param,ndim,sigma, obskernel = observations[oparam]

            mod_v=mean_velocity(stars,rlower,rmid, rupper, param, ndim, kernel=obskernel,**kwargs)


            #Compare density profiles
            mindx=(mod_v > 0.) * (rupper < 1.e10)
            pyplot.loglog(rmid[mindx],mod_v[mindx],'r',label='Model')
            pyplot.loglog(rmid[mindx],mod_v[mindx],'ro')

            mindx=(v > 0.) * (rupper < 1.e10)

            pyplot.loglog(rmid[mindx],v[mindx],'k',label='Observations')
            pyplot.loglog(rmid[mindx],v[mindx],'ko')

            pyplot.legend()
            pyplot.xlabel(r'$\log_{10} r$ (pc)')

            pyplot.ylabel(r'$ \log_{10} <%s>$ ($\rm km/s$)' % oparam)

            if filename is not None:

                if nv == 0:
                    pyplot.savefig(filename)
                else:
                    dotspot=filename.find('.')
                    newfilename = filename[:dotspot]+('_%i' % nv)+filename[dotspot:]
                    pyplot.savefig(newfilename)

                pyplot.close()

            else:
                pyplot.show()
                pyplot.close()


            nv+=1

