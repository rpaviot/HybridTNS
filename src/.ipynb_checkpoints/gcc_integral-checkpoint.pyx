#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport exp,cos,sin



cdef size_t size_W=500
#cdef size_t size_GL=300

cdef double EPSABS=1e-10
cdef double EPSREL=1e-5

cdef int NDIM=2
cdef int NVEC=1
cdef int VERBOSE=0
cdef int LAST=4
cdef int SEED=0
cdef int MINEVAL=0
cdef int MAXEVAL=10000
#cdef FLATNESS NULL
#cdef SPIN NULL
cdef int NNEW=10000
cdef int NMIN=2
cdef double FLATNESS=40
cdef size_t ncomp = 1
cdef size_t neval = 0
                    



cdef double complex gamma_c(double complex x) nogil:
    cdef double complex gamma 
    cdef gsl_sf_result norm
    cdef gsl_sf_result theta
    gsl_sf_lngamma_complex_e(creal(x), cimag(x), &norm,&theta)
    gamma = exp(norm.val)*cos(theta.val) + I*exp(norm.val)*sin(theta.val) 
    return gamma
    

#def fgamma(z):
#    cdef double complex zc = <double complex>z
#    cdef double result = gamma_c(zc)
#    return result

cdef double int_gsl_qags(double func(double, void *) nogil, void * p, double xmin, double xmax) nogil:
    cdef double result, error;
    cdef gsl_integration_workspace * W
    cdef gsl_function F
    W = gsl_integration_workspace_alloc(size_W)
    F.function = func
    F.params = p
    gsl_integration_qags(&F, xmin, xmax,EPSABS, EPSREL, size_W, W, &result, &error)    
    gsl_integration_workspace_free(W)
    return result


cdef double int_gsl_qag(double func(double, void *) nogil, void * p, double xmin, double xmax) nogil:
    cdef double result, error;
    cdef gsl_integration_workspace * W
    cdef gsl_function F
    W = gsl_integration_workspace_alloc(size_W)
    F.function = func
    F.params = p
    gsl_integration_qag(&F, xmin, xmax,EPSABS, EPSREL, size_W, GSL_INTEG_GAUSS21,W, &result, &error)    
    gsl_integration_workspace_free(W)
    return result


cdef double int_gsl_qng(double func(double, void *) nogil, void * p, double xmin, double xmax) nogil:
    cdef double result, error;
    cdef gsl_function F
    F.function = func
    F.params = p
    gsl_integration_qng(&F, xmin, xmax,EPSABS, EPSREL, &result, &error,&neval)    
    return result



cdef double int_gsl_GaussLegendre(double func(double, void *) nogil, void * p, double xmin, double xmax, size_t n) nogil:
    cdef double result, error
    cdef gsl_integration_glfixed_table * W 
    cdef gsl_function F
    W = gsl_integration_glfixed_table_alloc(n)
    F.function = func
    F.params = p
    result = gsl_integration_glfixed(&F, xmin, xmax, W)    
    gsl_integration_glfixed_table_free(W)
    return result


    

cdef double int_gsl_cquad(double func(double, void *) nogil, void * p, double xmin, double xmax) nogil:
    cdef double result, error;
    cdef gsl_integration_cquad_workspace * W
    cdef gsl_function F
    W = gsl_integration_cquad_workspace_alloc(size_W)
    F.function = func    
    F.params = p
    gsl_integration_cquad(&F, xmin, xmax, EPSABS, EPSREL, W, &result, &error,NULL)    
    gsl_integration_cquad_workspace_free(W)
    return result




# cdef void int_suave(cubareal* result, integrand_t func, void *p) nogil:
#     cdef int nregions, neval, fail 
#     error = <cubareal *>malloc(ncomp * sizeof(cubareal))
#     prob =  <cubareal *>malloc(ncomp * sizeof(cubareal))

#     Suave(NDIM, ncomp, func, p, NVEC,
#         EPSREL, EPSABS, VERBOSE, SEED,
#         MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,
#         NULL, NULL,
#         &nregions, &neval, &fail, result, error, prob)
    
#     free(error)
#     free(prob)
                    
        