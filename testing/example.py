#!/usr/bin/env python
# coding: utf-8

# In[2]:


import m2mcluster as m2m


# In[7]:


from amuse.lab import *
from amuse.units import nbody_system,units

import matplotlib.pyplot as plt
import numpy as np


# In[8]:


from amuse.community.hermite.interface import Hermite


# In[ ]:


mmin=0.1 | units.MSun
mmax=1.4 | units.MSun


# In[9]:


ocluster,oconverter=m2m.setup_star_cluster(N=1000, Mcluster = 1000 | units.MSun, Rcluster = 9. |units.parsec, W0=7.,imf='kroupa',mmin=mmin, mmax=mmax)


# In[4]:


#Setup a star cluster for artificial observation
ocluster,oconverter=m2m.setup_star_cluster(N=1000, Mcluster = 1000 | units.MSun, Rcluster = 9. |units.parsec, W0=7.)

#Measure artifical cluster's density profile asssuming an equal number of stars per bin
orlower,orad,orupper,orho=m2m.density(ocluster,bins=True,bintype='fix')
orlower[0]=0.
print(orho)


# In[5]:


while np.sum(orho==0.)!=0:
    ocluster,oconverter=m2m.setup_star_cluster(N=1000, Mcluster = 1000 | units.MSun, Rcluster = 9. |units.parsec, W0=7.)
    orlower,orad,orupper,orho=m2m.density(ocluster,bins=True,bintype='fix')
    orlower[0]=0.


# In[6]:


sigv=m2m.velocity_dispersion(ocluster,rlower=orlower,rmid=orad,rupper=orupper)


# In[7]:


#Initialize an M2M Star cluster
#Specify number of iterations to run algorithm for
#Specify number of workers to be used by Nbody code
cluster=m2m.starcluster(number_of_iterations=500,number_of_workers=1,debug=False)


# In[8]:


#Add the "observed" cluster density profile as an observable
cluster.add_observable(orlower,orad,orupper,orho,'density',ndim=3)
cluster.add_observable(orlower,orad,orupper,sigv,'velocity',ndim=3)


# In[9]:


#Initialize a model star cluster will an initial guess as the observed cluster's properties
cluster.initialize_star_cluster(N=1000, Mcluster = 1000 | units.MSun, Rcluster = 3. |units.parsec, softening = 0.01 | units.parsec, W0=1.)


# In[10]:


#Plot initial positions
cluster.xy_plot(filename='xyplot0.png')


# In[11]:


#Compare initial density profiles
cluster.rho_prof(filename='rhoplot0.png')


# In[12]:


cluster.sigv_prof()


# In[13]:


#Exectute the made to measure algorithm
outfile=open('logfile','w')
for i in range(0,cluster.number_of_iterations):
    #Initialize a new N-body simulation ever time step. 
    #In this example I use 1% of the cluster's dynamical time for the integration timeste
    cluster.initialize_gravity_code('Hermite', dt=0.01*cluster.tdyn, theta=0.6)
    #Evolve the model cluster forward for 10% of its dynamical time
    cluster.evolve(tend=0.1*cluster.tdyn)
    #Run the M2M algorithm, which will adjust all of the stellar masses based on kernel function
    cluster.evaluate(kernel=None,epsilon=1.,mu=1.,alpha=1.,plot=False)
    #Compare the new model density profile to the observed one
    cluster.rho_prof(filename='%s.png' % str(i).zfill(5))
    #Centre the star cluster and find determine Nbody conversion scales for next integration
    cluster.reinitialize_star_cluster(mmin= mmin, mtot=1000.0 | units.MSun)

    print(i,len(cluster.stars))
    
    outfile.write('%i %f\n' % (i,cluster.criteria))
    
outfile.close()


# In[14]:


cluster.rho_prof()


# In[15]:


cluster.sigv_prof()


# In[16]:


cluster.xy_plot()


# In[ ]:





# In[ ]:




