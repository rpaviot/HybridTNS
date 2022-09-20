#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True
import numpy as np

cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport exp
from main cimport *


cdef class sp:
    def __cinit__(self):
        
        global acc,spline
        acc[0]  = gsl_interp_accel_alloc ()
        spline[0]  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline[0],&kc[0],&Pklin[0],Npo)
    
        acc[1]  = gsl_interp_accel_alloc ()
        spline[1]  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline[1],&kc[0],&Pdd[0],Npo)
        
        acc[2]  = gsl_interp_accel_alloc ()
        spline[2]  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline[2],&kc[0],&Pdt[0],Npo)
        
        acc[3]  = gsl_interp_accel_alloc ()
        spline[3]  = gsl_spline_alloc(gsl_interp_cspline,Npo)
        gsl_spline_init (spline[3],&kc[0],&Ptt[0],Npo)
    
        acc[4]  = gsl_interp_accel_alloc ()
        spline[4]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[4],&kcorr[0],&Id2[0],Ncorr)
        
        acc[5]  = gsl_interp_accel_alloc ()
        spline[5]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[5],&kcorr[0],&Ig2[0],Ncorr)
        
        acc[6]  = gsl_interp_accel_alloc ()
        spline[6]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[6],&kcorr[0],&Fg2[0],Ncorr)
        
        acc[7]  = gsl_interp_accel_alloc ()
        spline[7]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[7],&kcorr[0],&Id2d2[0],Ncorr)
        
        acc[8]  = gsl_interp_accel_alloc ()
        spline[8]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[8],&kcorr[0],&Id2g2[0],Ncorr)
        
        acc[9]  = gsl_interp_accel_alloc ()
        spline[9]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[9],&kcorr[0],&Ig2g2[0],Ncorr)
        
        acc[10]  = gsl_interp_accel_alloc ()
        spline[10]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[10],&kcorr[0],&Id2theta[0],Ncorr)
        
        acc[11]  = gsl_interp_accel_alloc ()
        spline[11]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[11],&kcorr[0],&Ig2theta[0],Ncorr)
        
        acc[12]  = gsl_interp_accel_alloc ()
        spline[12]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[12],&kcorr[0],&Fg2theta[0],Ncorr)
    
        acc[13]  = gsl_interp_accel_alloc ()
        spline[13]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[13],&kcorr[0],&A11[0],Ncorr)
        
        acc[14]  = gsl_interp_accel_alloc ()
        spline[14]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[14],&kcorr[0],&A12[0],Ncorr)
        
        acc[15]  = gsl_interp_accel_alloc ()
        spline[15]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[15],&kcorr[0],&A22[0],Ncorr)
    
        acc[16]  = gsl_interp_accel_alloc ()
        spline[16]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[16],&kcorr[0],&A23[0],Ncorr)
        
        acc[17]  = gsl_interp_accel_alloc ()
        spline[17]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[17],&kcorr[0],&A33[0],Ncorr)
        
        acc[18]  = gsl_interp_accel_alloc ()
        spline[18]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[18],&kcorr[0],&B111[0],Ncorr)
    
        acc[19]  = gsl_interp_accel_alloc ()
        spline[19]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[19],&kcorr[0],&B112[0],Ncorr)
        
        acc[20]  = gsl_interp_accel_alloc ()
        spline[20]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[20],&kcorr[0],&B121[0],Ncorr)
        
        acc[21]  = gsl_interp_accel_alloc ()
        spline[21]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[21],&kcorr[0],&B122[0],Ncorr)
                   
        acc[22]  = gsl_interp_accel_alloc ()
        spline[22]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[22],&kcorr[0],&B211[0],Ncorr)
    
        acc[23]  = gsl_interp_accel_alloc ()
        spline[23]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[23],&kcorr[0],&B212[0],Ncorr)
        
        acc[24]  = gsl_interp_accel_alloc ()
        spline[24]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[24],&kcorr[0],&B221[0],Ncorr)
        
        acc[25]  = gsl_interp_accel_alloc ()
        spline[25]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[25],&kcorr[0],&B222[0],Ncorr)
            
        acc[26]  = gsl_interp_accel_alloc ()
        spline[26]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[26],&kcorr[0],&B312[0],Ncorr)
        
        acc[27]  = gsl_interp_accel_alloc ()
        spline[27]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[27],&kcorr[0],&B321[0],Ncorr)
        
        acc[28]  = gsl_interp_accel_alloc ()
        spline[28]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[28],&kcorr[0],&B322[0],Ncorr)
        
        acc[29]  = gsl_interp_accel_alloc ()
        spline[29]  = gsl_spline_alloc(gsl_interp_cspline,Ncorr)
        gsl_spline_init (spline[29],&kcorr[0],&B422[0],Ncorr)   
        
    @staticmethod
    cdef double Pk_lin(double k) nogil:
        if ((k < KMIN) or ( k >KMAX)):
            return 0
        else:
            return gsl_spline_eval(spline[0],k, acc[0]) 
            
    @staticmethod
    cdef double Pk_dd(double k) nogil:
        if ((k < KMIN) or ( k >KMAX)):
            return 0
        else:
            return gsl_spline_eval(spline[1],k, acc[1])     
        
        
    @staticmethod            
    cdef double Pk_dt(double k) nogil:
        if ((k < KMIN) or ( k >KMAX)):
            return 0
        else:
            return gsl_spline_eval(spline[2],k, acc[2])  
    
    @staticmethod            
    cdef double Pk_tt(double k) nogil:
        if ((k < KMIN) or ( k >KMAX)):
            return 0
        else:
            return gsl_spline_eval(spline[3],k, acc[3])   


        
    @staticmethod
    cdef double Pk_Id2(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[4],k,acc[4])
        
    @staticmethod   
    cdef double Pk_Ig2(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[5],k,acc[5])
        
    @staticmethod           
    cdef double Pk_Fg2(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[6],k,acc[6])   
        
    @staticmethod            
    cdef double Pk_Id2d2(double k) nogil:
        if (k < KMINcorr):
            return 0 
        elif (k > KMAXcorr):
            return gsl_spline_eval(spline[7],KMAXcorr,acc[7])#*exp(-k)
        else :
            return gsl_spline_eval(spline[7],k,acc[7])#*exp(-k)
        
    @staticmethod            
    cdef double Pk_Id2g2(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[8],k,acc[8])
        
    @staticmethod            
    cdef double Pk_Ig2g2(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[9],k,acc[9])
        
        
    @staticmethod            
    cdef double Pk_Id2theta(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[10],k,acc[10])

    @staticmethod
    cdef double Pk_Ig2theta(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[11],k,acc[11])
        
        
    @staticmethod           
    cdef double Pk_Fg2theta(double k) nogil:
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else :
            return gsl_spline_eval(spline[12],k,acc[12])   
        

    @staticmethod            
    cdef double cA(double k,int m,int n) nogil:
    
        cdef int ind=m*10+n;
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else:    
            if ind == 11:
                return gsl_spline_eval (spline[13], k, acc[13])
            elif ind == 12:
                return gsl_spline_eval (spline[14], k, acc[14])
            elif ind == 22:
                return gsl_spline_eval (spline[15], k, acc[15])
            elif ind == 23:
                return gsl_spline_eval (spline[16], k, acc[16])
            elif ind == 33:
                return gsl_spline_eval (spline[17], k, acc[17])
            else:
                return 0
        
    
    @staticmethod        
    cdef double cB(double k,int m,int n,int o) nogil:
    
        cdef int ind=m*100+n*10+o
        if ((k < KMINcorr) or ( k >KMAXcorr)):
            return 0
        else:
            if ind == 111:
                return gsl_spline_eval (spline[18], k, acc[18])
            elif ind == 112:
                return gsl_spline_eval (spline[19], k, acc[19])
            elif ind == 121:
                return gsl_spline_eval (spline[20], k, acc[20])
            elif ind == 122:
                return gsl_spline_eval (spline[21], k, acc[21])
            elif ind == 211:
                return gsl_spline_eval (spline[22], k, acc[22])
            elif ind == 212:
                return gsl_spline_eval (spline[23], k, acc[23])
            elif ind == 221:
                return gsl_spline_eval (spline[24], k, acc[24])
            elif ind == 222:
                return gsl_spline_eval (spline[25], k, acc[25])
            elif ind == 312:
                return gsl_spline_eval (spline[26], k, acc[26])
            elif ind == 321:
                return gsl_spline_eval (spline[27], k, acc[27])
            elif ind == 322:
                return gsl_spline_eval (spline[28], k, acc[28])
            elif ind == 422:
                return gsl_spline_eval (spline[29], k, acc[29])
            else:
                return 0
        

cdef class free_splines:
    def __cinit__(self):
        pass
    def __dealloc__(self):
        gsl_spline_free (spline[0])
        gsl_interp_accel_free (acc[0])
        
        gsl_spline_free (spline[1])
        gsl_interp_accel_free (acc[1])    
        
        gsl_spline_free (spline[2])
        gsl_interp_accel_free (acc[2])    
        
        gsl_spline_free (spline[3])
        gsl_interp_accel_free (acc[3]) 
        
        gsl_spline_free (spline[4])
        gsl_interp_accel_free (acc[4])   
        
        gsl_spline_free (spline[5])
        gsl_interp_accel_free (acc[5])  
        
        gsl_spline_free (spline[6])
        gsl_interp_accel_free (acc[6]) 
        
        gsl_spline_free (spline[7])
        gsl_interp_accel_free (acc[7])  
        
        gsl_spline_free (spline[8])
        gsl_interp_accel_free (acc[8])  
        
        gsl_spline_free (spline[9])
        gsl_interp_accel_free (acc[9])  
        
        gsl_spline_free (spline[10])
        gsl_interp_accel_free (acc[10])  
    #         
        gsl_spline_free (spline[11])
        gsl_interp_accel_free (acc[11])  
        
        gsl_spline_free (spline[12])
        gsl_interp_accel_free (acc[12]) 
       
        gsl_spline_free (spline[13])
        gsl_interp_accel_free (acc[13]) 
        
        gsl_spline_free (spline[14])
        gsl_interp_accel_free (acc[14]) 
        
        gsl_spline_free (spline[15])
        gsl_interp_accel_free (acc[15])   
        
        gsl_spline_free (spline[16])
        gsl_interp_accel_free (acc[16]) 
        
        gsl_spline_free (spline[17])
        gsl_interp_accel_free (acc[17])  
        
        gsl_spline_free (spline[18])
        gsl_interp_accel_free (acc[18])  
        
        gsl_spline_free (spline[19])
        gsl_interp_accel_free (acc[19])  
        
        gsl_spline_free (spline[20])
        gsl_interp_accel_free (acc[20]) 
        
        gsl_spline_free (spline[21])
        gsl_interp_accel_free (acc[21])    
        
        gsl_spline_free (spline[22])
        gsl_interp_accel_free (acc[22])    
        
        gsl_spline_free (spline[23])
        gsl_interp_accel_free (acc[23])    
        
        gsl_spline_free (spline[24])
        gsl_interp_accel_free (acc[24])    
        
        gsl_spline_free (spline[25])
        gsl_interp_accel_free (acc[25])    
            
        gsl_spline_free (spline[26])
        gsl_interp_accel_free (acc[26])    
        
        gsl_spline_free (spline[27])
        gsl_interp_accel_free (acc[27])       
      
        gsl_spline_free (spline[28])
        gsl_interp_accel_free (acc[28])    
        
        gsl_spline_free (spline[29])
        gsl_interp_accel_free (acc[29])              
            
 
            