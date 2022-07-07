import numpy as np

def get_kernel(r,rlower,rmid,rupper,kernel='identifier',ndim=3,**kwargs):

    vol=(4./3.)*np.pi*(rupper**3.-rlower**3.)
    area=np.pi*(rupper**2.-rlower**2.)

    knorm=kwargs.get('knorm',True)


    if kernel == 'identifier':
        rindx=((r >= rlower) * (r <= rupper))

        #D. Syer & S. Tremaine 1996 - Section 2.2, set K_j to 1/vol or 0, such that K_j/Z_j = 1 or 0
        if ndim==3:
            if knorm:
                K_j=rindx*(1./vol)
            else:
                K_j=rindx*1.
        elif ndim==2:
            if knorm:
                K_j=rindx*(1./area) 
            else:
                K_j=rindx*1.

    elif kernel == 'gaussian':

        #D. Syer & S. Tremaine 1996 - Section 2.2 - use bin centre as mean of Gaussian and sigma of 1/2 bin width
        ksigma=kwargs.get('ksigma',(rupper-rlower)/2.)

        rindx=((r >= rlower) * (r <= rupper))

        if np.sum(rindx)!=0:
            rmean=rmid[rindx][0]
            sig=ksigma[rindx][0]
            K_j=gaussian_kernel(rmid,rmean,sig)
        else:
            K_j=np.zeros(len(rlower))

        if ndim==3 and knorm:
            K_j=K_j*(1./vol)
        elif ndim==2 and knorm:
            K_j=K_j*(1./area)
        else:
            K_j*=1.

    elif kernel == 'epanechnikov':

        rindx=((r >= rlower) * (r <= rupper))

        if np.sum(rindx)!=0:
            h=kwargs.get('h',(rupper[rindx][0]-rlower[rindx][0]))
            x=np.fabs(rmid-r)
            K_j=epanechnikov_kernel(x,h)
        else:
            K_j=np.zeros(len(rlower))

        if ndim==3 and knorm:
            K_j=K_j*(1./vol)
        elif ndim==2 and knorm:
            K_j=K_j*(1./area)
        else:
            K_j*=1.

    elif kernel == 'sph':
        h=kwargs.get('h',(rupper-rlower))

        if isinstance(h,float):
            h=np.ones(len(rupper))*h

        rindx=((r >= rlower) * (r <= rupper))

        if np.sum(rindx)!=0:
            x=np.fabs(rmid-r)
            K_j=sph_kernel(x,h[rindx])

        else:
            K_j=np.zeros(len(rlower))

        if ndim==3 and knorm:
            K_j=K_j*(1./vol)
        elif ndim==2 and knorm:
            K_j=K_j*(1./area)
        else:
            K_j*=1.

    return K_j


def gaussian_kernel(r,rmean,sigma):

    sigma2=sigma**2.
    f=(1./np.sqrt(2*np.pi*sigma2))*np.exp(-0.5*((r-rmean)**2.)/sigma2)

    f/=np.sum(f)

    return f

def epanechnikov_kernel(x,h):

    f=np.zeros(len(x))
    xindx=(x>=0) * (x<=h)
    f[xindx]=(3./(4.*h))*(1.0-(x[xindx]/h)**2.)

    return f

def bovy_epanechnikov_kernel(r,h):
    out= numpy.zeros_like(r)
    out[(r >= 0.)*(r <= h)]= 3./4.*(1.-r[(r >= 0.)*(r <= h)]**2./h**2.)/h
    return out

def sph_kernel(x,h):
    f=np.zeros(len(x))
    

    xindx=(x/h>=0) * (x/h<=0.5)
        
    if np.sum(xindx)>0:
        f[xindx]=1.0-6.0*((x[xindx]/h)**2.)+6.0*((x[xindx]/h)**3.)
    
    xindx=(x/h>=0.5) * (x/h<=1.)

    if np.sum(xindx)>0:
        f[xindx]=2.0*(1.0-((x[xindx]/h)**3.))
        
    f*=8.0/(np.pi*(h**3.))
    
    return f

