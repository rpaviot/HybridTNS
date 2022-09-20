#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True
import numpy as np
cimport numpy as np
from gcc_integral cimport *
from main cimport *
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt,exp,log,M_PI,pow
from cython.parallel cimport prange


cdef :
    gsl_interp_accel * acc_pklin
    gsl_spline * spline_pklin    
    
    gsl_interp_accel * acc_ptt
    gsl_spline * spline_ptt   
    
    gsl_interp_accel * acc_pdt
    gsl_spline * spline_pdt
    

cdef class compute:
        
    def __cinit__(self):
        global acc_pklin,spline_pklin
        global acc_ptt,spline_ptt
        global acc_pdt,spline_pdt

        acc_pklin  = gsl_interp_accel_alloc ()
        spline_pklin  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline_pklin,&kc[0],&Pklin[0],Npo)
        
        acc_ptt  = gsl_interp_accel_alloc ()
        spline_ptt  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline_ptt,&kc[0],&Ptt[0],Npo)
        
        acc_pdt  = gsl_interp_accel_alloc ()
        spline_pdt  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline_pdt,&kc[0],&Pdt[0],Npo)
        
                
        self.get_corrections()
        
        
    cdef void get_corrections(self):
        global A11,A12,A22,A23,A33,B111,B112,B121,B122,B211,B212,B221,B222,B312,B321,B322,B422 

        cdef Py_ssize_t i
        arrt = np.zeros((17,Ncorr),dtype=np.float64)
        A11,A12,A22,A23,A33,B111,B112,B121,B122,B211,B212,B221,B222,B312,B321,B322,B422  = np.ascontiguousarray(arrt)
        
        
        for i in prange(0,Ncorr,nogil=True):
            A11[i] =  integral_corrA(kcorr[i],1,1) 
            A12[i] =  integral_corrA(kcorr[i],1,2) 
            A22[i] =  integral_corrA(kcorr[i],2,2) 
            A23[i] =  integral_corrA(kcorr[i],2,3) 
            A33[i] =  integral_corrA(kcorr[i],3,3) 
            B111[i] =  integral_corrB(kcorr[i],1,1,1) 
            B112[i] =  integral_corrB(kcorr[i],1,1,2) 
            B121[i] = integral_corrB(kcorr[i],1,2,1) 
            B122[i] =  integral_corrB(kcorr[i],1,2,2) 
            B211[i] =  integral_corrB(kcorr[i],2,1,1) 
            B212[i] =  integral_corrB(kcorr[i],2,1,2) 
            B221[i] =  integral_corrB(kcorr[i],2,2,1) 
            B222[i] =  integral_corrB(kcorr[i],2,2,2) 
            B312[i] =  integral_corrB(kcorr[i],3,1,2) 
            B321[i] =  integral_corrB(kcorr[i],3,2,1) 
            B322[i] =  integral_corrB(kcorr[i],3,2,2) 
            B422[i] =  integral_corrB(kcorr[i],4,2,2) 

                    

    def __dealloc__(self):
        gsl_spline_free (spline_pklin)
        gsl_interp_accel_free (acc_pklin)       
        gsl_spline_free (spline_ptt)
        gsl_interp_accel_free (acc_ptt) 
        gsl_spline_free (spline_pdt)
        gsl_interp_accel_free (acc_pdt)
  
            

cdef double Pk(double k) nogil:
    if ((k < KMIN) or ( k >KMAX)):
        return 0
    else:
        return gsl_spline_eval(spline_pklin,k, acc_pklin) 
    

cdef double Pktt(double k) nogil:
    if ((k < KMIN) or ( k >KMAX)):
        return 0
    else:
        return gsl_spline_eval(spline_ptt,k, acc_ptt) 
    

cdef double Pkdt(double k) nogil:
    if ((k < KMIN) or ( k >KMAX)):
        return 0
    else:
        return gsl_spline_eval(spline_pdt,k, acc_pdt) 
                



cdef double Ax(double r,double x,int m, int n) nogil:
    
    cdef int ind = m*10+n
    
    if ind == 11:
        return (-r*r*r/7.0)*(x+6*x*x*x+r*r*x*(-3.0+10*x*x)+r*(-3.0+x*x-12*x*x*x*x))
    elif ind == 12:
        return r*r*r*r/14.0*(x*x-1.0)*(-1.0+7*r*x-6*x*x)
    elif ind == 22:
        return r*r*r/14.0*(r*r*x*(13.0-41*x*x)-4*(x+6*x*x*x)+r*(5.0+9*x*x+42*x*x*x*x))
    elif ind == 23:
        return r*r*r*r/14.0*(x*x-1.0)*(-1.0+7*r*x-6*x*x)
    elif ind == 33:
        return r*r*r/14.0*(1.0-7*r*x+6*x*x)*(-2*x+r*(-1.0+3*x*x))
    else:
        return 0



cdef double Atx(double r,double x,int m,int n) nogil:
    
    cdef int ind = m*10+n
    
    if ind == 11:
            return (1.0/7.0)*(x+r-2*r*x*x)*(3*r+7*x-10*r*x*x)
    elif ind == 12:
            return r/14.0*(x*x-1.0)*(3*r+7*x-10*r*x*x)
    elif ind == 22:
            return 1.0/14.0*(28*x*x+r*x*(25.0-81*x*x)+r*r*(1.0-27*x*x+54*x*x*x*x))
    elif ind == 23:
            return r/14.0*(1.0-x*x)*(r-7*x+6*r*x*x)
    elif ind == 33:
            return 1.0/14.0*(r-7*x+6*r*x*x)*(-2*x-r+3*r*x*x)
    else:
        return 0



cdef double ax(double r, double x,int m,int n) nogil:

    cdef int ind = m*10+n

    if ind == 11:
            return (1.0/7.0)*(-7.0*x*x+r*r*r*x*(-3.0+10.0*x*x)+3.0*r*(x+6.0*x*x*x)+r*r*(6.0-19.0*x*x-8.0*x*x*x*x))
    elif ind == 12:
            return r/14.0*(-1.0+x*x)*(6.0*r-7.0*(1.0+r*r)*x+8.0*r*x*x)
    elif ind == 22:
            return 1.0/14.0*(-28.0*x*x+r*r*r*x*(-13.0+41.0*x*x)+r*x*(11.0+73.0*x*x)-2.0*r*r*(-9.0+31.0*x*x+20.0*x*x*x*x))
    elif ind == 23:
            return r/14.0*(-1.0+x*x)*(6.0*r-7.0*(1.0+r*r)*x+8.0*r*x*x)
    elif ind == 33:
            return 1.0/14.0*(7.0*x+r*(-6.0+7.0*r*x-8.0*x*x))*(-2.0*x+r*(-1.0+3.0*x*x))
    else :
        return 0

                                     
                                   
                                    
                          
cdef double B(double r,double x,int m, int n , int o) nogil:

    cdef int ind =m*100+n*10+o 

    if ind == 111:
        return r*r/2.0*(x*x-1)
    elif ind == 112:
        return 3*r*r/8.0*(x*x-1)*(x*x-1)
    elif ind == 121:
        return 3*r*r*r*r/8.0*(x*x-1)*(x*x-1)
    elif ind == 122:
        return 5*r*r*r*r/16.0*(x*x-1)*(x*x-1)*(x*x-1)
    elif ind == 211:
        return r/2.0*(r+2*x-3*r*x*x)
    elif ind == 212:
        return -3.0/4.0*r*(x*x-1)*(-r-2*x+5*r*x*x)
    elif ind == 221:
        return 3.0/4.0*r*r*(x*x-1)*(-2+r*r+6*r*x-5*r*r*x*x)
    elif ind == 222:
        return -3.0/16.0*r*r*(x*x-1)*(x*x-1)*(6-30*r*x-5*r*r+35*r*r*x*x)
    elif ind == 312:
        return r/8.0*(4*x*(3-5*x*x)+r*(3-30*x*x+35*x*x*x*x))
    elif ind == 321:
        return r/8.0*(-8*x+r*(-12+36*x*x+12*r*x*(3-5*x*x)+r*r*(3-30*x*x+35*x*x*x*x)))
    elif ind == 322:
        return 3.0/16.0*r*(x*x-1)*(-8*x+r*(-12+60*x*x+20*r*x*(3-7*x*x)+5*r*r*(1-14*x*x+21*x*x*x*x)))
    elif ind == 422:
        return r/16.0*(8*x*(-3+5*x*x)-6*r*(3-30*x*x+35*x*x*x*x)+6*r*r*x*(15-70*x*x+63*x*x*x*x)+r*r*r*(5-21*x*x*(5-15*x*x+11*x*x*x*x)))
    else:
        return 0

cdef double integrand_corrA(double x, double r, double k, int m, int n) nogil:
    cdef double l=sqrt(1.0+r*r-2*r*x)
    return Ax(r,x,m,n)*Pk(k)*Pk(k*l)/(l*l*l*l)

cdef double integrand_corrAt(double x, double r, double k,int m, int n) nogil:
    cdef double l=sqrt(1.0+r*r-2*r*x)
    return Atx(r,x,m,n)*Pk(k*r)*Pk(k*l)/(l*l*l*l)

cdef double integrand_corra(double x, double r, double k, int m, int n) nogil:
    cdef double l=sqrt(1.0+r*r-2*r*x)
    return ax(r,x,m,n)*Pk(k*r)*Pk(k)/(l*l)





cdef double integrandx_corrA(double x, void * p) nogil:
    cdef double r = (<double *> p)[0]
    cdef double k = (<double *> p)[1]
    cdef double m = (<double *> p)[2]
    cdef double n = (<double *> p)[3]
    cdef int m1 = (int)(m)
    cdef int n1 = (int)(n)

    return integrand_corrA(x,r,k,m1,n1)

cdef double integrandx_corrAt(double x, void * p) nogil:
    cdef double r = (<double *> p)[0]
    cdef double k = (<double *> p)[1]
    cdef double m = (<double *> p)[2]
    cdef double n = (<double *> p)[3]
    cdef int m1 = (int)(m)
    cdef int n1 = (int)(n)
    
    return integrand_corrAt(x,r,k,m1,n1)

cdef double integrandx_corra(double x, void * p) nogil:
    cdef double r = (<double *> p)[0]
    cdef double k = (<double *> p)[1]
    cdef double m = (<double *> p)[2]
    cdef double n = (<double*> p)[3]
    cdef int m1 = (int)(m)
    cdef int n1 = (int)(n)
    
    return integrand_corra(x,r,k,m1,n1)


cdef double integrandr_corrA(double r, void *p) nogil:
    cdef double * params = <double *> malloc(4*sizeof(double)) 
    cdef double k = (<double *> p)[0]
    cdef double m = (<double *> p)[1]
    cdef double n = (<double *> p)[2]
    cdef double result
    r=exp(r);
    params[0] = r
    params[1] = k
    params[2] = m
    params[3] = n
    result = int_gsl_qag(integrandx_corrA, params,-1.,1.)
    free(params)
    return result*r

cdef double integrandr_corrAt(double r, void *p) nogil:
    cdef double * params = <double *> malloc(4*sizeof(double)) 
    cdef double k = (<double *> p)[0]
    cdef double m = (<double *> p)[1]
    cdef double n = (<double *> p)[2]
    cdef double result
    r=exp(r);
    params[0] = r
    params[1] = k
    params[2] = m
    params[3] = n
    result = int_gsl_qag(integrandx_corrAt, params,-1.,1.)
    free(params)
    return result*r


cdef double integrandr_corra(double r, void *p) nogil:
    cdef double * params = <double *> malloc(4*sizeof(double)) 
    cdef double k = (<double *> p)[0]
    cdef double m = (<double *> p)[1]
    cdef double n = (<double *> p)[2]
    cdef double result
    r=exp(r);
    params[0] = r
    params[1] = k
    params[2] = m
    params[3] = n
    result = int_gsl_qag(integrandx_corra,params,-1.,1.)
    free(params)
    return result*r



cdef double integral_corrA(double k,double m, double n) nogil:
    cdef double * params = <double *> malloc(3*sizeof(double)) 
    cdef size_t W
    cdef double result,result2,result3

    params[0]=k
    params[1]=m
    params[2]=n

    if (k > 60):
        result = 0
        result2 = 0
        result3 = 0
    else : 
        result = int_gsl_qag(integrandr_corrA,params,log(KMIN),log(KMAX))
        result2 = int_gsl_qag(integrandr_corrAt, params,log(KMIN),log(KMAX))
        result3 = int_gsl_qag(integrandr_corra, params,log(KMIN),log(KMAX))
        
    free(params)
    return (result+result2+result3)/(2*2*M_PI*M_PI)*(k*k*k)




cdef double integrand_corrB(double x, double r, double k,double m,double n,double o) nogil:
    cdef double pk1, pk2
    cdef double l=sqrt(1.0+r*r-2*r*x);
    
    cdef int m1 = (int)(m)
    cdef int n1 = (int)(n)
    cdef int o1 = (int)(o)
    
    #printf("%i \n",n)

    #if (n1==1):
    #    pk1=Pkdt(k*l)
    #else:
    #    pk1=Pktt(k*l)
    #if (o1==1):
    #    pk2=Pkdt(k*r) 
    #else:
    #    pk2=Pktt(k*r)
    pk1=Pk(k*l);
    pk2=Pk(k*r);


    return B(r,x,m1,n1,o1)*pk1*pk2*pow(l,-2*n1);


cdef double integrandx_corrB(double x, void *p) nogil:
    cdef double r = (<double *> p)[0]
    cdef double k = (<double *> p)[1]
    cdef double m = (<double *> p)[2]
    cdef double n = (<double *> p)[3]
    cdef double o = (<double *> p)[4]
    return integrand_corrB(x,r,k,m,n,o)

cdef double integrandr_corrB(double r, void *p) nogil:
    cdef double k = (<double *> p)[0]
    cdef double m = (<double *> p)[1]
    cdef double n = (<double *> p)[2]
    cdef double o = (<double *> p)[3]
    cdef double * params = <double *> malloc(5*sizeof(double)) 
    cdef double result

    r=exp(r);
    params[0] = r
    params[1] = k
    params[2] = m
    params[3] = n
    params[4] = o
    result = int_gsl_qag(integrandx_corrB, params,-1.,1.)
    free(params)
    return result*r

cdef double integral_corrB(double k,double m,double n,double o) nogil:
    
    cdef double * params = <double *> malloc(4*sizeof(double)) 
    cdef double result
    cdef size_t W

    params[0]=k
    params[1]=m
    params[2]=n
    params[3]=o
    
    if (k > 60):
        result = 0
    else :
        result = int_gsl_qag(integrandr_corrB, params,log(KMIN),log(KMAX))
        
    free(params)
    return result/(2*2*M_PI*M_PI)*(k*k*k)

    
    

# cdef int cuba_integrand_B(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *p) nogil:
    
#     cdef double x = xx[0]
#     cdef double r = xx[1]
#     cdef double k = (<double *> p)[0]
    
#     cdef double xp = -1 + 2*x
#     cdef double rp = KMIN*exp(log(KMAX/KMIN)*r)
#     cdef double result
    
#     result = 2*log(KMAX/KMIN)*integrand_corrB(xp,k,rp)
#     ff[0] = result*rp/(2*2*M_PI*M_PI)*(k*k*k)
    
    



# cdef double integral_corrB_cuba(double k,int m, int n,int o) nogil:
#     global M,N,O
#     M=m
#     N=n
#     O=o
#     cdef cubareal result
#     int_suave(&result, cuba_integrand_B, &k)
#     return result


# cdef int cuba_integrand_A(const int *ndim, const cubareal xx[], const int *ncomp, cubareal ff[], void *p) nogil:
    
#     cdef double x = xx[0]
#     cdef double r = xx[1]
#     cdef double k = (<double *> p)[0]
    
#     cdef double xp = -1 + 2*x
#     cdef double rp = KMIN*exp(log(KMAX/KMIN)*r)
#     result = 2*log(KMAX/KMIN)*(integrand_corrA(xp,k,rp) + integrand_corrAt(xp,k,rp) + integrand_corra(xp,k,rp))
#     ff[0] = result*rp/(2*2*M_PI*M_PI)*(k*k*k)
    

# cdef double integral_corrA_cuba(double k,int m, int n) nogil:
#     cdef cubareal result
#     int_suave(&result, cuba_integrand_A, &k)
#     return result

    
    
    
    
                         
    



