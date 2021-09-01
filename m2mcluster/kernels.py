import numpy as np

def get_kernel(r,rlower,rmid,rupper,kernel='identifier',ndim=3,**kwargs):

    vol=(4./3.)*np.pi*(rupper**3.-rlower**3.)
    area=(4.)*np.pi*(rupper**2.-rlower**2.)


    if kernel == 'identifier':
        rindx=((r >= rlower) * (r <= rupper))

        #D. Syer & S. Tremaine 1996 - Section 2.2, set K_j to 1/vol or 0, such that K_j/Z_j = 1 or 0
        if ndim==3:
            K_j=rindx*(1./vol) 
        elif ndim==2:
            K_j=rindx*(1./area) 

    elif kernel == 'gaussian':

        #D. Syer & S. Tremaine 1996 - Section 2.2 - use bin centre as mean of Gaussian and sigma of 1/2 bin width
        ksigma=kwargs.get('ksigma',(rupper-rlower)/2.)

        rindx=((r >= rlower) * (r <= rupper))

        if np.sum(rindx)!=0:
            rmean=rmid[rindx][0]
            sig=ksigma[rindx][0]
            K_j=gaussian_kernel(rmean,rmean,sig)
        else:
            K_j=np.zeros(len(rlower))

        if ndim==3:
            K_j=K_j*(1./vol)
        elif ndim==2:
            K_j=K_j*(1./area)


    return K_j


def gaussian_kernel(r,rmean,sigma):

	sigma2=sigma**2.
	f=(1./np.sqrt(2*np.pi*sigma2))*np.exp(-0.5*((r-rmean)**2.)/sigma2)
	
	return f
