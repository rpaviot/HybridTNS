import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"./src")
import HybridTNS

#import TNScorr
import time
from mcfit import P2xi
#import cosmo.py 
#import integral_utils (intégration routine)
# Entrée Cosmology, Measure, Covariance
# Crée une classe cosmology. 
#linear_power = pycamb.Cosmo()
#print(linear_power.pars)
#k,pk = linear_power.get_pklin()
#func = interpolate.CubicSpline(k,pk)


params = {'h':0.67,'Omega_cdm':0.319-0.049, 'Omega_b':0.049,'YHe':0.24,'T_cmb':2.7255,
          'N_eff':3.046,'N_ncdm':0,'n_s':0.960, 'A_s':2.0e-09,'s8':0.83,
          'output': 'mPk','z_pk':0.58, 'gauge':'synchroton'}

#params = {'h':0.7,'Omega_cdm':0.239, 'Omega_b':0.047,'YHe':0.24,'T_cmb':2.7255,
#          'N_eff':3.046,'N_ncdm':0,'n_s':0.960, 'A_s':2.14681e-09,'s8':None,
#          'output': 'mPk','P_k_max_h/Mpc': '200.0','z_pk':0.55, 'gauge':'synchroton'}


model = HybridTNS.model(params,lmax=4,FoG_type="Kurtosis",saveout=True,filepath="./")

aper = 1.0
apar = 1.0
f = 0.8
b1 = 2
b2 = 0
bg2 = 0
bT3 = 0
sigmaV = 4.1
at = 1

s = np.linspace(0.5,199.5,200)
dictp = model.pk_multipoles(aper, apar, f, b1, b2, bg2, bT3, sigmaV, at)
#print(dictp)
dictm = model.xi_multipoles(s=s)

rr,DeltaSigma = model.DeltaSigma2(b1,b2,bg2,bT3)
rr2,DeltaSigma2 = model.DeltaSigma2(b1,b2,bg2,bT3)


#rp1x,rp2x,rp3x,wpdata,err = np.loadtxt("/Users/rpaviot/Downloads/wgg_CMASS_flagship_treecorr.dat",unpack=True)
#_,wpdata2 = np.loadtxt("/Users/rpaviot/Downloads/wgg_CMASS_flagship_corrfunc.dat",unpack=True)

#rpbins = np.logspace(np.log10(1e-1),np.log10(100),22)
#aper = 1.0
#apar = 1.0
#f = 0.78
#b1 = 1.715
#b2 = -0.2
#bg2 = -(3/7.)*(b1 - 1.)
#bT3 = (22./42.)*(b1 - 1.)
#sigmaV = 2
#at = 0
#rpt,wp = model.wgg(aper, apar, f, b1, b2, bg2, bT3, sigmaV,at,rp=rp2x,pimax=100)
#rpt,wp2 = model.wgg(aper, apar, f, b1, b2, bg2, bT3,sigmaV,at,rp=rp2x,pimax=50)
#plt.show()
#plt.plot(s,dictp['p2']*s**2)
#plt.plot(s,dictp['p4']*s**2)

#real.save_output()

#cosmo.HybridTNS()
#cosmo.set_sigma8(0.8)
#k,pk = cosmo.get_linear_pk()
#cosmo.Hybrid_TNS() 
#cosmo.get_TNScorr()
#print(k,pk)



#time_in = time.time()
#result = integrate.quad(func, 1e-4,50)
#time_out = time.time()
#print(time_out - time_in)
#print(result)

#TNScorr.init_pklin(k,pk)

#print(integral,integral2)
