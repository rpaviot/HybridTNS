#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import respresso
import sys
import scipy.integrate
import corrTNS
import os
import os.path

from extrap import pk_extrapolate
from classy import Class
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline as CS
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import UnivariateSpline as us

from scipy import integrate
from pybiasterm import NL_bias


class RealSpace:
    """Compute Real space quantities
    The non-linear matter power spectrum is computed with the respresso formalism (https://arxiv.org/abs/1708.08946)
    The velocity power spectra are computed with Bel et al. fitting functions (https://arxiv.org/abs/1809.09338)
    The non linear bias terms are computed with the formalism described in (https://arxiv.org/abs/1708.08130)"""
    
    def __init__(self, params,
                 Npo,kmin=1e-5, kmax=100.0,kcorr = np.logspace(np.log10(1e-4),np.log10(100),256),Om0=0.31):
        
        params['P_k_max_h/Mpc']=kmax
        self.k = np.logspace(np.log10(kmin),np.log10(kmax),Npo,dtype=np.float64)
        self.Npo = Npo
        self.kcorr = kcorr
        self.zeff = params['z_pk']
        self.params = params
        self.data = {}
        
        """ Class params dict """
        
        self.class_params = dict(self.params)
        del self.class_params['s8']
        self.Om0 = Om0

        cosmo = Class()
        self.cosmo = cosmo

        ratio = 1
        if params['s8'] is not None :
            print("Rescaling pklin given the input s8")
            pklin = self.linear_power_spectrum(self.class_params,z=0)
            self.spline_Pklin = CS(self.k,pklin)
            sigma8 = self.get_sigmaR(8)
            self.sigma8 = sigma8
            sigma8_input = params['s8']
            ratio = (self.sigma8/sigma8_input)**2
            self.class_params['A_s']= self.class_params['A_s']/ratio
            
                    
        
    def linear_power_spectrum(self, params,z):

        k = self.k
        self.cosmo.set(params)
        self.cosmo.compute()

        Pk_lin = np.zeros(len(self.k),dtype=np.float64)
        Pk_lin = np.array([self.cosmo.pk(k[i]*self.cosmo.h(), z)*self.cosmo.h()**3 for i in range(0,len(self.k))],dtype=np.float64)
        return Pk_lin

    
    def set_pklin(self,k,pk):
        self.spline_Pklin = CS(self.k,pk)
        

    @staticmethod
    def THwindow(x):
        return 3*(np.sin(x)-x*np.cos(x))/x**3

    def integrand(self,k,r):
        return k**2 * self.spline_Pklin(k)/(2.*np.pi**2) * self.THwindow(k*r)**2

    def get_sigmaR(self,r):
        s8 = np.sqrt(integrate.quad(self.integrand, 1e-5,10,args=(8))[0])
        return s8

    def set_global_params(self,params,z):
        pars = {}
        pars['Om0']=self.Om0
        pars['z'] = z
        pars['D_A'] = self.cosmo.angular_distance(z)
        pars['D_M'] = pars['D_A']*(1.+z)
        pars['h'] = self.params['h']
        pars['H_z'] = self.cosmo.Hubble(z)
        pars['D_H'] = 1./pars['H_z']
        pars['D_V'] = (pars['z']*(pars['D_M'])**2*pars['D_H'])**(1/3)
        pars['r_drag'] = self.cosmo.rs_drag()
        pars['DM_rd'] = pars['D_M']/pars['r_drag']
        pars['DH_rd'] = pars['D_H']/pars['r_drag']
        pars['DV_rd'] = pars['D_V']/pars['r_drag']

        sigma8 = self.get_sigmaR(8)
        self.sigma8 = sigma8
        pars['f'] = self.cosmo.scale_independent_growth_factor_f(z)
        pars['s8'] = sigma8
        pars['A_s'] = params['A_s']
        pars['n_s']= params['n_s']
        pars['fs8'] = pars['f']*pars['s8']
        self.pars = pars
    
 
    def HF_power_spectrum(self, params,z):
        cosmo = self.cosmo

        class_params = dict(params)
        class_params['non linear']='halofit'

        self.cosmo.set(class_params)
        cosmo.compute()


        Pk_HF = np.zeros(len(self.k))
        Pk_HF = np.array([self.cosmo.pk(self.k[i]*self.cosmo.h(), z)*self.cosmo.h()**3 for i in range(0,len(self.k))],dtype=np.float64)
        return Pk_HF
            
        
        
    def respresso_power_spectrum(self,Pk_lin,Pk_HF):
        respresso_obj = respresso.respresso_core()
        plin_target_spl = ius(self.k,Pk_lin)
        respresso_obj.set_target(plin_target_spl)
        respresso_obj.find_path()
        kwave = respresso_obj.get_kinternal()
        Pnl = respresso_obj.reconstruct()

        kmax = respresso_obj.get_kmax()

        spline_Pnl = us(kwave,Pnl)
        spline_PHF = ius(self.k,Pk_HF)


        ## MATCHING HIGH K respresso amplitude to HALOFIT one.

        cond = np.where((kwave > kmax)&(kwave < kmax + 0.3))
        pfit = Pnl[cond]
        pnl = spline_PHF(kwave[cond])

        scaling,_ = curve_fit(self.linear_scaling,pnl,pfit)
        Pdd = np.zeros(self.k.size,dtype=np.float64)
        condl = self.k < min(kwave)
        condh = self.k > max(kwave)
        condm = (self.k > min(kwave)) & (self.k < max(kwave))
        Pdd[condl] = self.spline_Pklin(self.k[condl])
        Pdd[condm] = spline_Pnl(self.k[condm])
        Pdd[condh] = scaling*spline_PHF(self.k[condh])

        return Pdd
        
 
    def linear_scaling(self,data,b):
        return b*data

    def velocity_power_spectrum(self,Pk_lin):
        a1 = -0.817 + 3.198*self.sigma8
        a2 = 0.877 - 4.191*self.sigma8
        a3 = -1.199 + 4.629*self.sigma8
        return Pk_lin*np.exp(-self.k*(a1 + a2*self.k + a3*self.k**2))

    def cross_power_spectrum(self,Pk_lin,Pdd):
        k0 = -0.017 + 1.496*self.sigma8**2
        ba = 0.091 + 0.702*self.sigma8**2
        return pow(Pk_lin*Pdd,1./2)*np.exp(-self.k*k0 - ba*self.k**6)       
        
    def set_galaxy_power_spectra(self):
        
        Pk_lin = self.linear_power_spectrum(self.class_params,self.zeff)
        self.set_pklin(self.k,Pk_lin)
        self.set_global_params(self.class_params,self.zeff)
        
        Pk_HF =self.HF_power_spectrum(self.class_params,self.zeff)
        Pdd = self.respresso_power_spectrum(Pk_lin,Pk_HF)       
        Ptt = self.velocity_power_spectrum(Pk_lin)
        Pdt = self.cross_power_spectrum(Pk_lin,Pdd)
        
        extrapol = pk_extrapolate()
        k,Pk_lin,Pdd,Pdt,Ptt =  extrapol.get_pk(ns=self.pars['n_s'],k=self.k,pklin=Pk_lin,pdd=Pdd,pdt=Pdt,ptt=Ptt)
        self.k = k 

        self.data['k']=self.k
        self.data['Pk_lin'] = Pk_lin
        #self.data['Pk_HF'] = Pk_HF
        self.data['Pdd'] = Pdd
        self.data['Pdt'] = Pdt
        self.data['Ptt'] = Ptt
        
        NLbias = NL_bias(self.k, self.data['Pk_lin'],self.kcorr)
        Id2, Ig2, Fg2, Id2theta, Ig2theta, Fg2theta, Id2d2, Id2g2, Ig2g2 = NLbias()
        self.data['kcorr'] = self.kcorr
        self.data['Id2']= np.array(Id2)
        self.data['Ig2'] = np.array(Ig2)
        self.data['Fg2'] = np.array(Fg2)
        self.data['Id2theta'] = np.array(Id2theta)
        self.data['Ig2theta'] = np.array(Ig2theta)
        self.data['Fg2theta'] = np.array(Fg2theta)
        self.data['Id2d2'] = np.array(Id2d2)
        self.data['Id2g2']= np.array(Id2g2)
        self.data['Ig2g2']= np.array(Ig2g2)

            
    def get_attributes(self):
        return self.data,self.pars
        
    
       


class model(RealSpace):
    def __init__(self, params,lmax=4,FoG_type="Lorentzian",Npo=512,saveout=None,filepath=None):

        self.Om0 = params['Omega_cdm']+params['Omega_b']
        self.Npo = Npo
        self.filename="pks_h_{}_Om_{}_zeff_{}_Npo_{}.npy".format(params['h'],self.Om0,params['z_pk'],self.Npo)
        self.filepath = filepath
        self.get_filepath()

        if os.path.exists(self.filename):
            self.read_calc()         
            model = corrTNS.TNS(self.theory,self.cosmoparams,lmax,FoG_type)

        else:
            super().__init__(params,Npo,Om0=self.Om0)
            super().set_galaxy_power_spectra()
            self.theory, self.cosmoparams = super().get_attributes()
            model = corrTNS.TNS(self.theory,self.cosmoparams,lmax,FoG_type)
            A11,A12,A22,A23,A33,B111,B112,B121,B122,B211,B212,B221,B222,B312,B321,B322,B422 = model.get_corr()

            self.theory['cosmo']= self.cosmoparams
            self.theory['A11'] = np.array(A11)
            self.theory['A12'] = np.array(A12)
            self.theory['A22'] = np.array(A22)
            self.theory['A23'] = np.array(A23)
            self.theory['A33'] = np.array(A33)
            self.theory['B111'] = np.array(B111)
            self.theory['B112'] = np.array(B112)
            self.theory['B121'] = np.array(B121)
            self.theory['B122'] = np.array(B122)
            self.theory['B211'] = np.array(B211)
            self.theory['B212'] = np.array(B212)
            self.theory['B221'] = np.array(B221)
            self.theory['B222'] = np.array(B222)
            self.theory['B312'] = np.array(B312)
            self.theory['B321'] = np.array(B321)
            self.theory['B322'] = np.array(B322)
            self.theory['B422'] = np.array(B422)

        if saveout is True:
            self.save_calc()

        self.model = model
        self.init()
        print("TNS model ready")
        
        
    def pk_multipoles(self,*args,**qwargs):
        p = self.model.pk_multipoles(*args,**qwargs)
        return p
    
    def xi_multipoles(self,*args,**qwargs):
        xi = self.model.xi_multipoles(*args,**qwargs)
        return xi
    
    def wgg(self,*args,**qwargs):
        rp,wp = self.model.wgg(*args,**qwargs)
        return rp,wp
    
    def init(self):
        self.model.init_model()

    def free(self):
        self.model.free_model()

    def get_filepath(self):
        if self.filepath is None :
            self.filename = "./" + self.filename
        else:
            self.filename = self.filepath + self.filename

    def read_calc(self):
        self.theory = np.load(self.filename,allow_pickle='TRUE').item()
        self.cosmoparams = self.theory['cosmo']
        
    def save_calc(self):
        np.save(self.filename, self.theory)

    def get_pks(self):
        return self.theory

    def get_cosmo(self):
        return self.cosmoparams



        
        
        
        
        
        
        
        

        
