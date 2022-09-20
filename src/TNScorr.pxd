#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True
#cdef double integral_corrA_cuba(double k,int m, int n) nogil
#cdef double integral_corrB_cuba(double k,int m, int n, int o) nogil
cdef class compute:
    #def __cinit__(self,double [::1] k,double [::1] pklin,double [::1])
    cdef void get_corrections(self)    

    
cdef extern from "math.h":
    cdef double fabs (double x) nogil
    cdef double pow (double x, double y) nogil
    cdef double exp(double x) nogil
