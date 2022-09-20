#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True

cdef extern from "complex.h":
    cdef double creal(double complex z) nogil
    cdef double cimag(double complex z) nogil
    cdef double complex ctan(double complex x) nogil
    cdef double complex cpow(double complex x,double complex y) nogil
    cdef double complex I 
    
cdef extern from "gsl/gsl_sf_result.h":
  ctypedef struct gsl_sf_result:
    double val
    double err
    
    
cdef extern from "gsl/gsl_sf_gamma.h":
    int  gsl_sf_lngamma_complex_e(double zr, double zi, gsl_sf_result * lnr, gsl_sf_result * arg) nogil
    

cdef extern from "gsl/gsl_math.h":

  ctypedef struct gsl_function:
    double (* function) (double x, void * params) nogil
    void * params

cdef extern from "gsl/gsl_integration.h":

    ctypedef struct gsl_integration_workspace
    ctypedef struct gsl_integration_qaws_table
    ctypedef struct  gsl_integration_qawo_table
    ctypedef struct gsl_integration_cquad_workspace
    ctypedef struct gsl_integration_glfixed_table
    cdef enum:
        GSL_INTEG_GAUSS15 = 1
        GSL_INTEG_GAUSS21 = 2
        GSL_INTEG_GAUSS31 = 3
        GSL_INTEG_GAUSS41 = 4
        GSL_INTEG_GAUSS51 = 5
        GSL_INTEG_GAUSS61 = 6
  

    gsl_integration_workspace *  gsl_integration_workspace_alloc(const size_t n) nogil
    gsl_integration_cquad_workspace *  gsl_integration_cquad_workspace_alloc(const size_t n) nogil
    gsl_integration_glfixed_table * gsl_integration_glfixed_table_alloc(size_t n) nogil 


    int gsl_integration_cquad (const gsl_function * f, double a, double b, double epsabs, double epsrel, gsl_integration_cquad_workspace * workspace, double * result, double * abserr, size_t * nevals) nogil
    int  gsl_integration_qag(const gsl_function *f, double a, double b, double epsabs, double epsrel, size_t limit, int key, gsl_integration_workspace * workspace, double * result, double * abserr) nogil
    int gsl_integration_qng(const gsl_function *f, double a, double b, double epsabs, double epsrel, double *result, double *abserr, size_t *neval) nogil
    int  gsl_integration_qags(const gsl_function * f, double a, double b, double epsabs, double epsrel, size_t limit, gsl_integration_workspace * workspace, double *result, double *abserr) nogil
    double gsl_integration_glfixed(const gsl_function *f, double a, double b, gsl_integration_glfixed_table * t) nogil

    void  gsl_integration_cquad_workspace_free (gsl_integration_cquad_workspace * w) nogil
    void  gsl_integration_workspace_free(gsl_integration_workspace * w) nogil 
    void  gsl_integration_glfixed_table_free(gsl_integration_glfixed_table *t) nogil    
    
cdef extern from "gsl/gsl_interp.h":
  
  ctypedef struct gsl_interp_accel
  ctypedef struct gsl_interp_type
  ctypedef struct gsl_interp

  gsl_interp_type * gsl_interp_cspline
  gsl_interp_accel * gsl_interp_accel_alloc() nogil  
  void gsl_interp_accel_free(gsl_interp_accel * a) nogil
    
  
cdef extern from "gsl/gsl_spline.h":
  ctypedef struct gsl_spline
  
  gsl_spline * gsl_spline_alloc(const  gsl_interp_type * T, size_t size) nogil
  int gsl_spline_init(gsl_spline * spline, const double xa[],const double ya[], size_t size) nogil
    
  double gsl_spline_eval( gsl_spline * spline, double x, gsl_interp_accel * a) nogil
  double gsl_spline_eval_deriv( gsl_spline * spline, double x, gsl_interp_accel * a) nogil
    
  void gsl_spline_free(gsl_spline * spline) nogil

    
    
cdef extern from "gsl/gsl_errno.h":
 ctypedef void gsl_error_handler_t
 int GSL_SUCCESS
 int GSL_EUNDRFLW
 char *gsl_strerror(int gsl_errno)
 gsl_error_handler_t* gsl_set_error_handler_off()


# cdef extern from "cuba.h":
    
#   ctypedef double cubareal;

#   ctypedef int (*integrand_t)(const int *ndim, const cubareal x[],
#   const int *ncomp, cubareal f[], void *userdata) nogil

#   ctypedef void (*peakfinder_t)(const int *ndim, const cubareal b[],
#   int *n, cubareal x[], void *userdata);

#   void Suave(const int ndim, const int ncomp,
#   integrand_t integrand, void *userdata, const int nvec,
#   const cubareal epsrel, const cubareal epsabs,
#   const int flags, const int seed,
#   const int mineval, const int maxeval,
#   const int nnew, const int nmin,
#   const cubareal flatness, const char *statefile, void *spin,
#   int *nregions, int *neval, int *fail,
#   cubareal integral[], cubareal error[], cubareal prob[]) nogil

#   void cubafork(void *pspin);
#   void cubawait(void *pspin);

#   void cubacores(const int *n, const int *p);
#   void cubaaccel(const int *n, const int *p);

#   void cubainit(void (*f)(), void *arg);
#   void cubaexit(void (*f)(), void *arg);


cdef double complex gamma_c(double complex x) nogil
cdef double int_gsl_qag(double func(double, void *) nogil, void * p, double xmin, double xmax) nogil
cdef double int_gsl_qng(double func(double, void *) nogil, void * p, double xmin, double xmax) nogil
cdef double int_gsl_qags(double func(double, void *) nogil, void *p, double xmin, double xmax) nogil
cdef double int_gsl_cquad(double func(double, void *) nogil, void * p, double xmin, double xmax) nogil
cdef double int_gsl_GaussLegendre(double func(double, void *) nogil, void * p, double xmin, double xmax, size_t n) nogil
#cdef void int_suave(cubareal* result, integrand_t func, void *p) nogil