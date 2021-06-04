#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as pyplot
from amuse.plot import scatter
import numpy 
from amuse.units import nbody_system,units
#from galpy.util import bovy_plot

from .functions import density

#import seaborn as sns
#df = sns.load_dataset('iris')

import numpy as np

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

def density_profile(stars,observation,filename=None):

    rlower,rupper,rho, param, ndim=observation
    rad=(rlower+rupper)/2.
    vol=(4./3.)*numpy.pi*(rupper**3.-rlower**3.)
    area=numpy.pi*(rupper**2.-rlower**2.)

    mod_rho=density(stars,rlower,rupper,ndim)

    #Compare density profiles
    mindx=(mod_rho > 0.)
    pyplot.plot(numpy.log10(rad[mindx]),numpy.log10(mod_rho[mindx]),'r',label='Model')
    pyplot.plot(numpy.log10(rad[mindx]),numpy.log10(mod_rho[mindx]),'ro')

    pyplot.plot(numpy.log10(rad),numpy.log10(rho),'k',label='Observations')
    pyplot.plot(numpy.log10(rad),numpy.log10(rho),'ko')

    pyplot.legend()
    pyplot.xlabel('$\log_{10} r$ (pc)')
    pyplot.ylabel(r'$\log_{10} \Sigma$ ($M_{\odot}/pc^2)$')

    if filename is not None:
        pyplot.savefig(filename)
        pyplot.close()
    else:
        pyplot.show()
        pyplot.close()

def density_plot(x,y,n_iteration,type):
    
    
    # Basic 2D density plot
    pyplot.hist2d(x, y, bins=(20, 20), cmap=pyplot.cm.jet)
    
    pyplot.xlabel('X (pc)')
    pyplot.ylabel('Y (pc)')

    pyplot.xlim(-0.01,0.01)
    pyplot.ylim(-0.01,0.01)
    
    if 'mod' in type:
        pyplot.title('Model')
    else:
        pyplot.title('Observations')

    if filename is not None:
        pyplot.savefig(filename)
        pyplot.close()
    else:
        pyplot.show()
        pyplot.close()
