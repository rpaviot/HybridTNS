#!/usr/bin/env python3
import numpy as np
from scipy.special import gamma
from joblib import Parallel,delayed
from scipy.interpolate import CubicSpline
#from numba import njit, prange,vectorize,complex128
import cmath 

class NL_bias:
    
    def ctan(self,x):
        func = np.vectorize(cmath.tan)
        return func(x)
    
    def __init__(self,k,pk,kcorr):
        self.k = k
        self.pk = pk
        self.kcorr = kcorr
        self.N = len(k)
        self.S_pk = CubicSpline(self.k,self.pk)
        self.pk_decomposition()
        self.compute_matrices()

    def I_loop(self,nu1,nu2):
        result = gamma(3./2. - nu1)*gamma(3./2. - nu2)*gamma(-3./2.+ nu1 + nu2)/(gamma(nu1)*gamma(nu2)*gamma(3 - nu1 - nu2))
        return 1./(8.*pow(np.pi,3./2.))*result

    def fM_Id2(self,nu1,nu2):
        result = ((3. - 2.*(nu1 + nu2))*(4. - 7.*(nu1 + nu2))/(14.*nu1*nu2))*self.I_loop(nu1,nu2);
        return result

    def fM_Id2theta(self,nu1,nu2):
        result = ((3. - 2.*(nu1 + nu2))*(8. - 7.*(nu1 + nu2))/(14.*nu1*nu2))*self.I_loop(nu1,nu2);
        return result    

    def fM_Ig2(self,nu1,nu2):
         result = -((3. - 2.*(nu1 + nu2))*(1 - 2.*(nu1 + nu2))*(6. + 7.*(nu1 + nu2))/(28.*nu1*(1.+nu1)*nu2*(1.+nu2))*self.I_loop(nu1,nu2));
         return result;    

    def fM_Ig2theta(self,nu1,nu2):
         result = -((3. - 2.*(nu1 + nu2))*(1 - 2.*(nu1 + nu2))*(-2. + 7.*(nu1 + nu2))/(28.*nu1*(1.+nu1)*nu2*(1.+nu2))*self.I_loop(nu1,nu2));
         return result;                                                   


    def fM_Fg2(self,nu1):
        result = -(15.*self.ctan(nu1*np.pi))/(28.*np.pi*(nu1+1.)*nu1*(nu1-1.)*(nu1-2.)*(nu1-3.));
        return result;                                                   

    def fM_Fg2theta(self,nu1):
         result = -(9.*self.ctan(nu1*np.pi))/(28.*np.pi*(nu1+1.)*nu1*(nu1-1.)*(nu1-2.)*(nu1-3.));
         return result;   


    def fM_Id2d2(self,nu1,nu2):
         result = 2*self.I_loop(nu1,nu2);
         return result;

    def fM_Ig2g2(self,nu1,nu2):
        result =((3. - 2.*(nu1 + nu2))*(1. - 2.*(nu1 + nu2)))/(nu1*(1.+nu1)*nu2*(1.+nu2))*self.I_loop(nu1,nu2);
        return result;    

    def fM_Id2g2(self,nu1,nu2):
         result = ((3. - 2*(nu1 + nu2))/(nu1*nu2))*self.I_loop(nu1,nu2);
         return result;  


    def f_Id2(self,k):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(np.dot(self.M_Id2,arr),arr)
        result = abs(k*k*k*np.real(product))
        return result

    def f_Ig2(self,k):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(np.dot(self.M_Ig2,arr),arr)
        result = abs(k*k*k*np.real(product))
        return result
    
    def f_Fg2(self,k,pk):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(self.M_Fg2,arr)
        result = abs(k*k*k*pk*np.real(product))
        return result

    def f_Id2theta(self,k):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(np.dot(self.M_Id2theta,arr),arr)
        result = abs(k*k*k*np.real(product))
        return result

    def f_Ig2theta(self,k):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(np.dot(self.M_Ig2theta,arr),arr)
        result = abs(k*k*k*np.real(product))
        return result
    
    def f_Fg2theta(self,k,pk):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(self.M_Fg2theta,arr)
        result = abs(k*k*k*pk*np.real(product))
        return result
    
    
    def f_Id2d2(self,k):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(np.dot(self.M_Id2d2,arr),arr)
        result = abs(k*k*k*np.real(product))
        return result

    def f_Id2g2(self,k):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(np.dot(self.M_Id2g2,arr),arr)
        result = abs(k*k*k*np.real(product))
        return result
 

    def f_Ig2g2(self,k):
        arr = self.cn*pow(k+0j,self.nm)
        product = np.dot(np.dot(self.M_Ig2g2,arr),arr)
        result = abs(k*k*k*np.real(product))
        return result
    

    def pk_decomposition(self): 
        bias = -1.600001
        delta = (1./(self.N-1.))*np.log(self.k[self.N-1]/self.k[0])

        cn_sym = np.zeros(self.N+1,dtype="complex128")
        indices = np.indices(cn_sym.shape)[0]
        mid = (int)(self.N/2)

        fft_in = self.pk*np.exp(-bias*indices[:-1]*delta)
        fft_out = np.fft.rfft(fft_in)

        nu_m = bias + (2*np.pi*1j/(self.N*delta))*(indices-mid)

        cond = indices - (int)(self.N/2) < 0
        indicesl = indices[cond]
        indicesh = indices[~cond]
        cn_sym[cond]=(1./self.N)*np.conj(fft_out[mid-indicesl])*pow((self.k[0]+0j),-nu_m[indicesl])
        cn_sym[~cond]=(1./self.N)*(fft_out[indicesh-mid])*pow((self.k[0]+0j),-nu_m[indicesh])
        cn_sym[0]=cn_sym[0]/2.
        cn_sym[self.N]=cn_sym[self.N]/2.
        
        self.cn = cn_sym
        self.nm = nu_m
        
        
    def compute_matrices(self):
        M_Id2 = np.zeros((self.N+1,self.N+1),dtype="complex128")
        M_Ig2 = np.zeros((self.N+1,self.N+1),dtype="complex128")
        M_Fg2 = np.zeros(self.N+1,dtype="complex128")
        M_Id2theta = np.zeros((self.N+1,self.N+1),dtype="complex128")
        M_Ig2theta = np.zeros((self.N+1,self.N+1),dtype="complex128")
        M_Fg2theta = np.zeros(self.N+1,dtype="complex128")
        M_Id2d2 = np.zeros((self.N+1,self.N+1),dtype="complex128")
        M_Id2g2 = np.zeros((self.N+1,self.N+1),dtype="complex128")
        M_Ig2g2 = np.zeros((self.N+1,self.N+1),dtype="complex128") 
        
        x,y = np.indices(M_Id2.shape)
        M_Id2[x,y] = self.fM_Id2(-self.nm[x]/2.,-self.nm[y]/2.)
        M_Ig2[x,y] = self.fM_Ig2(-self.nm[x]/2.,-self.nm[y]/2.)
        M_Fg2[x] = self.fM_Fg2(-self.nm[x]/2.)
        M_Id2theta[x,y] = self.fM_Id2theta(-self.nm[x]/2.,-self.nm[y]/2.)
        M_Ig2theta[x,y] = self.fM_Ig2theta(-self.nm[x]/2.,-self.nm[y]/2.)
        M_Fg2theta[x] = self.fM_Fg2theta(-self.nm[x]/2)
        M_Id2d2[x,y] = self.fM_Id2d2(-self.nm[x]/2.,-self.nm[y]/2.)
        M_Id2g2[x,y] = self.fM_Id2g2(-self.nm[x]/2.,-self.nm[y]/2.)
        M_Ig2g2[x,y] = self.fM_Ig2g2(-self.nm[x]/2.,-self.nm[y]/2.)
        
        self.M_Id2 = M_Id2
        self.M_Ig2 = M_Ig2
        self.M_Fg2 = M_Fg2
        self.M_Id2theta = M_Id2theta
        self.M_Ig2theta = M_Ig2theta
        self.M_Fg2theta = M_Fg2theta
        self.M_Id2d2 = M_Id2d2
        self.M_Id2g2 = M_Id2g2
        self.M_Ig2g2 = M_Ig2g2
        
        
    def __call__(self):
        
            
        Id2 = Parallel()(delayed(self.f_Id2)(self.kcorr[i]) for i in range(0,len(self.kcorr)))
        Ig2 = Parallel()(delayed(self.f_Ig2)(self.kcorr[i]) for i in range(0,len(self.kcorr)))
        Fg2 = Parallel()(delayed(self.f_Fg2)(self.kcorr[i],self.S_pk(self.kcorr[i])) for i in range(0,len(self.kcorr)))
        Id2theta = Parallel()(delayed(self.f_Id2theta)(self.kcorr[i]) for i in range(0,len(self.kcorr)))
        Ig2theta = Parallel()(delayed(self.f_Ig2theta)(self.kcorr[i]) for i in range(0,len(self.kcorr)))
        Fg2theta = Parallel()(delayed(self.f_Fg2theta)(self.kcorr[i],self.S_pk(self.kcorr[i])) for i in range(0,len(self.kcorr)))
        Id2d2 = Parallel()(delayed(self.f_Id2d2)(self.kcorr[i]) for i in range(0,len(self.kcorr)))
        Id2g2 = Parallel()(delayed(self.f_Id2g2)(self.kcorr[i]) for i in range(0,len(self.kcorr)))
        Ig2g2 = Parallel()(delayed(self.f_Ig2g2)(self.kcorr[i]) for i in range(0,len(self.kcorr)))
        Id2d2 = abs(Id2d2 - Id2d2[0])
        
        
        return Id2,Ig2,Fg2,Id2theta,Ig2theta,Fg2theta,Id2d2,Id2g2,Ig2g2