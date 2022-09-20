#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True
from gcc_integral cimport *


cdef extern from "stdio.h":
    int strcmp (const char* str1, const char* str2) nogil

cdef:
    const char* FoG_type
    const char* Lorentzian ="Lorentzian"
    const char* kurtosis ="kurtosis"
    double sigmaV_lin
    gsl_interp_accel * acc[30]	
    gsl_spline * spline[30]
    double [::1] kc
    double [::1] Pklin
    double [::1] Pdd
    double [::1] Pdt
    double [::1] Ptt
    double [::1] kcorr    
    double [::1] Id2
    double [::1] Ig2
    double [::1] Fg2
    double [::1] Id2d2
    double [::1] Id2g2 
    double [::1] Ig2g2
    double [::1] Id2theta
    double [::1] Ig2theta
    double [::1] Fg2theta
    double [::1] A11
    double [::1] A12
    double [::1] A22
    double [::1] A23
    double [::1] A33
    double [::1] B111
    double [::1] B112
    double [::1] B121
    double [::1] B122
    double [::1] B211
    double [::1] B212
    double [::1] B221
    double [::1] B222
    double [::1] B312
    double [::1] B321
    double [::1] B322
    double [::1] B422
    double KMIN
    double KMAX
    double KMINcorr
    double KMAXcorr
    int Npo
    int Ncorr
