import numpy as np

def get_kernel(r,rlower,rmid,rupper,kernel='identifier',ndim=3,**kwargs):

    vol=(4./3.)*np.pi*(rupper**3.-rlower**3.)
    area=np.pi*(rupper**2.-rlower**2.)

    knorm=kwargs.get('knorm',True)


    if kernel == 'identifier':
        rs=r.reshape(len(r),1)
        rmean_args=np.argmin(np.fabs(rs-rmid),axis=1)
        print(rmean_args)
        ids=np.arange(0,len(r))
        K_j=np.zeros((len(rmid),len(r)))
        K_j[rmean_args,ids]=1
        
        zargs=np.logical_or(r<rlower[0],r>rupper[-1])
        K_j[rmean_args[zargs],ids[zargs]]=0
        
        
        #D. Syer & S. Tremaine 1996 - Section 2.2, set K_j to 1/vol or 0, such that K_j/Z_j = 1 or 0
        if ndim==3 and knorm:
            K_j=K_j*(1./vol.reshape(len(vol),1))
        elif ndim==2 and knorm:
            K_j=K_j*(1./area.reshape(len(area),1))

    elif kernel == 'gaussian':

        #D. Syer & S. Tremaine 1996 - Section 2.2 - use bin centre as mean of Gaussian and sigma of 1/2 bin width
        ksigma=kwargs.get('ksigma',(rupper-rlower)/2.)

        rs=r.reshape(len(r),1)
        rmean_args=np.argmin(np.fabs(rs-rmid),axis=1)
        sig=ksigma[rmean_args]
        rmean=rmid[rmean_args]
        rmids=np.repeat(rmid,len(sig)).reshape(len(rmid),len(r))

        K_j=gaussian_kernel(rmids,rmean,sig)
        
        zargs=np.logical_or(r<rlower[0],r>rupper[-1])
        ids=np.arange(0,len(r))
        K_j[:,ids[zargs]]=0

        if ndim==3 and knorm:
            K_j=K_j*(1./vol.reshape(len(vol),1))
        elif ndim==2 and knorm:
            K_j=K_j*(1./area.reshape(len(area),1))
        elif ndim==3 and not knorm:

            vol=4.*np.pi*(np.tile(r,len(vol)).reshape((len(vol), len(r))))**2.
            K_j=K_j*(1./vol)


    elif kernel == 'loggaussian':

        vol=(4./3.)*np.pi*(rupper**3-rlower**3.)
        area=np.pi*(rupper**2.-rlower**2.)

        #lrupper=rupper #np.log10(rupper)
        #lrlower=rlower #np.log10(rlower)
        #lrmid=(lrupper+lrlower)/2.
        #lr=r #np.log10(r)

        lrupper=np.log(rupper)
        lrlower=np.log(rlower)
        lrmid=(lrupper+lrlower)/2.
        lr=np.log(r)

        #D. Syer & S. Tremaine 1996 - Section 2.2 - use bin centre as mean of Gaussian and sigma of 1/2 bin width
        lksigma=kwargs.get('ksigma',(lrupper-lrlower)/2.)

        lrs=lr.reshape(len(lr),1)
        lrmean_args=np.argmin(np.fabs(lrs-lrmid),axis=1)
        lsig=lksigma[lrmean_args]
        lrmean=lrmid[lrmean_args]
        lrmids=np.repeat(lrmid,len(lsig)).reshape(len(lrmid),len(lr))

        K_j=gaussian_kernel(lrmids,lrmean,lsig)
        
        zargs=np.logical_or(lr<lrlower[0],lr>lrupper[-1])
        ids=np.arange(0,len(lr))
        K_j[:,ids[zargs]]=0

        if ndim==3 and knorm:
            K_j=K_j*(1./vol.reshape(len(vol),1))
        elif ndim==2 and knorm:
            K_j=K_j*(1./area.reshape(len(area),1))
          
    return K_j

def gaussian_kernel(r,rmean,sigma):

    sigma2=sigma**2.
    f=(1./np.sqrt(2*np.pi*sigma2))*np.exp(-0.5*((r-rmean)**2.)/sigma2)

    f/=np.sum(f,axis=0)

    return f

