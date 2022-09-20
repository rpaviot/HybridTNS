#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True
import numpy as np
cimport numpy as np
from cspline cimport *
from gcc_integral cimport *
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt,exp,log,M_PI,pow

from astropy.cosmology import FlatLambdaCDM
#parsec in meters,M_sun in kg
from scipy.constants import parsec
from astropy import units as u
from libc.stdio cimport FILE, fopen, fread, fclose,printf


cdef :
    double rhom 
    double [::1] rr
    double [::1] rout
    double [::1] xi_gm
    double [::1] Sigma
    double [::1] Sigma_mean
    double [::1] Dsigmat
    gsl_interp_accel * acc_xigm
    gsl_spline * spline_xigm  
    gsl_interp_accel * acc_Sigma
    gsl_spline * spline_Sigma  
    int Npo,Np_out


cdef class compute:
    def __cinit__(self,params,r,xigm,rp=np.geomspace(1e-1,100,100)):
        global rr,xi_gm,rout,Dsigmat
        global acc_xigm,spline_xigm
        global acc_Sigma,spline_Sigma
        global Npo,Np_out
        global Sigma,Sigma_mean,Dsigmat


        self.get_meanrho(params)

        rr = np.ascontiguousarray(r)
        xi_gm = np.ascontiguousarray(xigm)
        rout = np.ascontiguousarray(rp)
    
        size = len(rout)
        Npo = <int>xi_gm.size
        Np_out = <int>size
        
        zero_array = np.zeros((3,size),dtype=np.float64)
        Sigma,Sigma_mean,Dsigmat = np.ascontiguousarray(zero_array)

        acc_xigm  = gsl_interp_accel_alloc ()
        spline_xigm  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline_xigm,&rr[0],&xi_gm[0],Npo)
    
        get_Sigma()

        acc_Sigma  = gsl_interp_accel_alloc ()
        spline_Sigma  = gsl_spline_alloc(gsl_interp_cspline,Np_out)
        gsl_spline_init (spline_Sigma,&rout[0],&Sigma[0],Np_out)

        get_Sigma_mean()

        cdef Py_ssize_t i
        for i in prange(0,Np_out,nogil=True):
            Dsigmat[i] = (Sigma_mean[i] - Sigma[i])/1e12


    def __dealloc__(self):
        gsl_spline_free (spline_xigm)
        gsl_interp_accel_free (acc_xigm)       
        gsl_spline_free (spline_Sigma)
        gsl_interp_accel_free (acc_Sigma)        

    def get_meanrho(self,params):
        global rhom
        Om0 = params['Om0']
        zeff = params['z']
        cosmo = FlatLambdaCDM(Om0=Om0,H0=100)
        #g/cm3
        meanrho = params['Om']*cosmo.critical_density(zeff).to(u.M_sun/u.Mpc**3).value
        rhom = meanrho

    def get_Dsigma(self):
        return rout,Dsigmat

cdef double int_Sigma(double chi,void * p) nogil:
    cdef double r  = (<double *> p)[0]
    dist = sqrt(pow(chi,2)+ pow(r,2))
    if (dist < rr[0]):
        return xi_gm[0]
    elif (dist > rr[Npo-1]):
        return 0
    else:
        return gsl_spline_eval(spline_xigm,dist,acc_xigm)

cdef double fSigma(double r) nogil:
    result = int_gsl_qag(int_Sigma,&r,0,1000)
    return result

cdef double int_Sigma_mean(double r,void * p) nogil:
    if (r < rout[0]):
        return r*Sigma[0]
    else:
        return r*gsl_spline_eval(spline_Sigma,r,acc_Sigma)

cdef double fSigma_mean(double r) nogil:
    result = int_gsl_qag(int_Sigma_mean,NULL,0,r)
    return result

cdef void get_Sigma() nogil:
    global Sigma
    cdef Py_ssize_t i
    for i in prange(0,Np_out,nogil=True):
        Sigma[i] = 2*fSigma(rout[i])*rhom

cdef void get_Sigma_mean() nogil:
    global Sigma_mean
    cdef Py_ssize_t i
    for i in prange(0,Np_out,nogil=True):
        Sigma_mean[i] = fSigma_mean(rout[i])*(2./pow(rout[i],2))







