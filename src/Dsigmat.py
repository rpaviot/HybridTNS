
### Not used ###
from scipy.integrate import quadrature as intg
from scipy.integrate import quad as intg2

import numpy as np 
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
"""parsec in meters,M_sun in kg"""
from scipy.constants import parsec
from astropy import units as u
import itertools

class model:
    def __init__(self,params):

        self.Om0 = params['Om0']
        zeff = params['z']
        cosmo = FlatLambdaCDM(Om0=self.Om0,H0=100)

        #g/cm3
        self.rhom = params['Om']*cosmo.critical_density(zeff).to(u.M_sun/u.Mpc**3).value

    def compute_DSigma(self,r,xi_gm,rp=np.geomspace(1e-1,100,100)):
        self.rp = rp
        self.spline_xi = interp1d(r, xi_gm,fill_value=(xi_gm[0],0),bounds_error=False)
        self.Sigma = np.array(list(map(self.fSigma,self.rp)))*self.rhom
        self.spline_Sigma = interp1d(self.rp,self.Sigma,fill_value=(self.Sigma[0],0),bounds_error=False)
        self.Sigma_mean = np.array(list(map(self.fSigma_mean,self.rp)))

        """M_sun / Mpc-2 to M_sun to pc-2"""
        self.Dsigma = (self.Sigma_mean - self.Sigma)/1e12
        self.spline_DSigma = interp1d(self.rp,self.Dsigma)
        return self.rp,self.Dsigma
    
    def int_Sigma(self,chi,rp): 
        return self.spline_xi(np.sqrt(chi**2+rp**2))

    def fSigma(self,rp):
        result,_ = intg2(self.int_Sigma,0,1000,args=rp)#,maxiter=200)
        return result

    def int_Sigma_mean(self,r):
        return r*self.spline_Sigma(r)

    def fSigma_mean(self,rp):
        result,_ = intg2(self.int_Sigma_mean,0,rp)#,maxiter=200)
        return result*(2./(rp**2))

    def DSigma_cut(self,r0):
        DSigma_cut = self.Dsigma - (r0/self.rp)**2*self.spline_DSigma(r0)
        return DSigma_cut

        
    
        



