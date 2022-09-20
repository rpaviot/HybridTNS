import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline as CS

class pk_extrapolate:
    
    def get_pk(self,kout=np.logspace(np.log10(1e-7),np.log10(1e4),512),**qwargs):
        self.ns = qwargs['ns']
        self.kout = kout
        self.k = qwargs['k']
        self.mink = min(self.k)
        self.maxk = max(self.k)
        self.CS_pklin = CS(self.k,qwargs['pklin'])
        self.pklin_min = self.CS_pklin(self.mink)
        self.CS_pdd = CS(self.k,qwargs['pdd'])
        self.CS_pdt = CS(self.k,qwargs['pdt'])
        self.CS_ptt = CS(self.k,qwargs['ptt'])
        
        fextrap_matter = np.vectorize(self.extrap_matter_power,excluded='self')
        fextrap_velocity = np.vectorize(self.extrap_velocity_power,excluded='self')
        self.fit_highk()
        pklin,pdd = fextrap_matter(self.kout)
        pdt,ptt = fextrap_velocity(self.kout)
        
        return kout,pklin,pdd,pdt,ptt
        
    def power_law(self,k,k0,alpha):
        return (k/k0)**(alpha)
        
    def fit_highk(self):
        kfit = np.geomspace(self.maxk/2,self.maxk,20)
        pklin_fit = self.CS_pklin(kfit)
        pk_fit = self.CS_pdd(kfit)
        self.params1,_ = curve_fit(self.power_law,kfit,pklin_fit,p0=[0.5,-1.5],bounds=([0.01,-5], [100,0]))
        self.params2,_ = curve_fit(self.power_law,kfit,pk_fit,p0=[0.5,-1.5],bounds=([0.01,-5], [100,0]))
    
    def extrap_matter_power(self,k):
        if (k < self.mink):
            pklin = self.pklin_min*(pow(k,self.ns)/pow(self.mink,self.ns))
            pdd = self.pklin_min*(pow(k,self.ns)/pow(self.mink,self.ns))
        elif ((k >= self.mink) & (k <= self.maxk)):
            pklin = self.CS_pklin(k)
            pdd = self.CS_pdd(k)
        elif (k > self.maxk):
            pklin = self.power_law(k,self.params1[0],self.params1[1])
            pdd = self.power_law(k,self.params2[0],self.params2[1])
        
        return pklin,pdd 
    
    def extrap_velocity_power(self,k):
        if (k < self.mink):
            pdt = self.pklin_min*(pow(k,self.ns)/pow(self.mink,self.ns))
            ptt = pdt
        elif ((k >= self.mink) & (k <= 5)):
            pdt = self.CS_pdt(k)
            ptt = self.CS_ptt(k)
        elif (k > 5):
            pdt = 0
            ptt = 0
        return pdt,ptt
            
        
        

#class xi_extrapolate:

 #       def get_xi(self,rout=np.logspace(np.log10(1e-8),np.log10(1e5),1024),**qwargs):
 #       self.rout = rout
 #       self.r = qwargs['r']
 #       self.minr = min(self.r)
 #       self.maxr = max(self.r)
 #       self.CS_pklin = CS(self.r,qwargs['xi'])
 #       self.pklin_min = self.CS_pklin(self.mink)
        
 #       fextrap_matter = np.vectorize(self.extrap_matter_power,excluded='self')
 #       fextrap_velocity = np.vectorize(self.extrap_velocity_power,excluded='self')
 #       self.fit_highk()
 #       pklin,pdd = fextrap_matter(self.kout)
 #       pdt,ptt = fextrap_velocity(self.kout)
        
 #       return kout,pklin,pdd,pdt,ptt
        



        
        