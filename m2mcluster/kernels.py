import numpy as np

def gaussian_kernel(r,rmean,sigma):

	sigma2=sigma**2.
	f=(1./np.sqrt(2*np.pi*sigma2))*np.exp(-0.5*((r-rmean)**2.)/sigma2)
	
	return f
