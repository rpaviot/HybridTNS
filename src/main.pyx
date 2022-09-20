#cython: language_level=3,boundscheck=False,wraparound=False,nonecheck=False,cdivision=True,profile=True
import numpy as np
from scipy.interpolate import CubicSpline as CS
from scipy.interpolate import UnivariateSpline as us
from scipy import integrate
from hankl import P2xi
import TNScorr


cimport numpy as np
from gcc_integral cimport *
from TNScorr cimport *
from main cimport *
from cspline cimport *
cimport Dsigma
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt,exp,log,M_PI




gsl_set_error_handler_off()

  
      
cdef:
    const char* FoG_type
    const char* Lorentzian ="Lorentzian"
    const char* kurtosis ="kurtosis"
    cdef int lmax
    double sigmaV_lin
    gsl_interp_accel * acc[30]	
    gsl_spline * spline[30]
    double [::1] kc
    double [::1] Pklin
    double [::1] Pdd
    double [::1] Pdt
    double [::1] Ptt
    double [::1] kcorr    
    double [::1] Id2
    double [::1] Ig2
    double [::1] Fg2
    double [::1] Id2d2
    double [::1] Id2g2 
    double [::1] Ig2g2
    double [::1] Id2theta
    double [::1] Ig2theta
    double [::1] Fg2theta
    double [::1] A11
    double [::1] A12
    double [::1] A22
    double [::1] A23
    double [::1] A33
    double [::1] B111
    double [::1] B112
    double [::1] B121
    double [::1] B122
    double [::1] B211
    double [::1] B212
    double [::1] B221
    double [::1] B222
    double [::1] B312
    double [::1] B321
    double [::1] B322
    double [::1] B422
    double KMIN
    double KMAX
    double KMINcorr
    double KMAXcorr
    int Npo
    int Ncorr
    double [::1] p0
    double [::1] p2
    double [::1] p4
    double [::1] p6
    double [::1] p8
    double [::1] Pgm
    int calc = 0

def L2(x):
    return (3.0*x*x-1.0)*0.5

def L4(x):
    return (35.0*x*x*x*x-30.0*x*x+3.0)/8.0;

    
cdef double Pl_2(double x) nogil:
    return (3.0*x*x-1.0)*0.5

cdef double Pl_4(double x) nogil:
    return (35.0*x*x*x*x-30.0*x*x+3.0)/8.0;

cdef double Pl_6(double x) nogil:
    return (231*pow(x,6) - 315*pow(x,4) + 105*x*x -5.0)/16.0;

cdef double Pl_8(double x) nogil:
    return (6435*pow(x,8)-12012*pow(x,6)+6930*pow(x,4)-1260*x*x+35)/128;
    
class TNS:


    def __init__(self,datain,cosmo,maxl,string_FoG,saveout=False,filepath=None):
        
        global Npo,KMIN,KMAX,Ncorr,KMINcorr,KMAXcorr
        global sigmaVlin,lmax
        global Pgm,p0,p2,p4,p6,p8
        global FoG_type
        
        lmax = maxl
        FoG_type = <bytes>string_FoG

        self.cosmo = cosmo
        self.kin = datain['k']
        
        self.read_python_input(datain)
    
        Npo = <int> kc.size
        Ncorr = <int> kcorr.size
        KMIN = kc[0]
        KMAX = kc[Npo-1]      
        KMINcorr = kcorr[0]
        KMAXcorr = kcorr[Ncorr - 1] 
        
        arrt = np.zeros((6,Npo),dtype=np.float64)
        Pgm,p0,p2,p4,p6,p8 = np.ascontiguousarray(arrt)
        sigmaV_lin = self.linear_sigmaV()

        
    def pk_multipoles(self, *args,**kwargs):
        global p0,p2,p4,p6,p8
        
        if kwargs:
            kin = kwargs['k']
            if ((np.min(kin)<KMIN) or (np.max(kin) > KMAX)):
                raise ValueError('Cannot extrapolate the model below kmin {} and above kmax {}'.format(KMIN,KMAX))   
            self.kin = kin
        else :        
            aper = <double>args[0]
            apar = <double>args[1]
            f = <double> args[2]
            b1 = <double>args[3]
            b2 = <double> args[4]
            bg2 = <double> args[5]
            bT3 = <double>args[6]
            FoGdamp = <double> args[7]
            
            TNS_multipoles(aper,apar,f,b1,b2,bg2,bT3,FoGdamp)
            outf = {}
            outf['k']=self.kin
            outf['p0']=np.array(p0)
            outf['p2']=np.array(p2)
            outf['p4']=np.array(p4)
            outf['p6']=np.array(p6)
            outf['p8']=np.array(p8)
            self.outf = outf
            return outf 

    
    def xi_multipoles(self,*args,**kwargs):
        global calc
        sbin = kwargs['s']
        self.sbin = sbin
        if not args and calc == 0:
            raise ValueError('You must provide a set of parameters')
        elif not args and calc == 1:
            out = self.HankelTransform(self.outf['p0'],self.outf['p2'],self.outf['p4'],self.outf['p6'],self.outf['p8'])
            return out
        elif args :
            self.pk_multipoles(*args)
            out = self.HankelTransform(self.outf['p0'],self.outf['p2'],self.outf['p4'],self.outf['p6'],self.outf['p8'])
            return out

    def xi_gm(self,*args,**kwargs):
        global Pgm
        cdef Py_ssize_t i

        #sbin = kwargs['s']
        b1 = <double>args[0]
        b2 = <double> args[1]
        bg2 = <double> args[2]
        bT3 = <double>args[3]

        for i in prange(0,Npo,nogil=True):
            Pgm[i] = Pk_gm(kc[i],b1,b2,bg2,bT3)
        
        pgm = np.array(Pgm)
        rr,xi_gm= P2xi(self.kin, pgm, l=0, n=0,lowring=True)
        return rr,xi_gm.real
        
    def DeltaSigma(self,*args,rp=np.geomspace(1e-1,100,100),r0=0.):
        rr,xi_gm = self.xi_gm(*args)
        modelD = Dsigma.compute(self.cosmo,rr,xi_gm,rp)
        rp,Dsigmat = modelD.get_Dsigma()
        rp = np.array(rp)
        Dsigmat = np.array(Dsigmat)
        spline = CS(rp,Dsigmat)
        if r0 == 0:
            return rp,Dsigmat
        else :
            Dsigma_cut = Dsigmat - (r0/rp)**2*spline(r0)     
            return rp,Dsigma_cut
    

    def wgg(self,*args,**kwargs):
        global calc
        rp = kwargs['rp']
        pimax = kwargs['pimax']
        
        dx = 0.01
        dpi = 1

        pibins = np.linspace(-pimax,pimax,2*int(pimax/dpi)+1)
        dxc = np.diff(pibins)[0];   pi = pibins[0:-1] + 0.5*dxc
        meshx,meshy = np.meshgrid(pi,rp,indexing='ij')
        r = np.sqrt(meshx**2 + meshy**2)
        mu = meshx/r

        maxdist = max(r.flatten())*sqrt(2)

        s = np.linspace(dx,maxdist+dx,int(maxdist/dx)+1)
        out = self.xi_multipoles(*args,s=s)
        xi0c = CS(out['s'],out['xi0'])
        xi2c = CS(out['s'],out['xi2'])
        xi4c = CS(out['s'],out['xi4'])

        xirppi = np.zeros((len(pi),len(rp)))
        x,y = np.indices(xirppi.shape)
        xirppi[x,y] = xi0c(r[x,y]) + xi2c(r[x,y])*L2(mu[x,y]) + \
        xi4c(r[x,y])*L4(mu[x,y])

        #dpi = 1
        #pibins = np.linspace(0,self.pimax,int(self.pimax/dpi)+1)
        #xirppi_new = stats.binned_statistic_2d(meshx.flatten(), meshy.flatten(), xirppi.flatten(), 'median', bins=[pibins, self.rpbins])

        self.wggr = np.sum(xirppi*dpi,axis = 0)
        return rp, self.wggr
    
    def linear_sigmaV(self):
        cdef double sigmaV
        spline = CS(kc,Pklin)
        result = integrate.fixed_quad(spline, 1e-4, 1)[0]
        sigmaV = np.sqrt(result/(6*np.pi**2))
        return sigmaV


    def carray(self,x):
        return np.ascontiguousarray(x)
    
    @staticmethod
    def fopen(filename):
        d = {}
        x= np.loadtxt(filename)    
        i,j = x.shape
        for j in range(0, j):
            d["arr{}".format(j)] = np.ascontiguousarray(x[:,j])
        return [d[field] for field in d.keys()]
    
    
    def init_model(self):
        """set global splines"""
        sp()

    def free_model(self):
        """Free global splines"""
        free_splines()
        
    def HankelTransform(self,p0,p2,p4,p6,p8):
        #rr, xi0 = P2xi(self.kin, p0, l=0, n=0,lowring=True)
        rr,xi0 = P2xi(self.kin, p0, l=0, n=0,lowring=True)
        rr,xi2 = P2xi(self.kin, p2, l=2, n=0,lowring=True)
        rr,xi4 = P2xi(self.kin, p4, l=4, n=0,lowring=True)
        rr,xi6 = P2xi(self.kin, p6, l=6, n=0,lowring=True)
        rr,xi8 = P2xi(self.kin, p8, l=8, n=0,lowring=True)
        
        xi0s = CS(rr,xi0.real)
        xi2s = CS(rr,xi2.real)
        xi4s = CS(rr,xi4.real)
        xi6s = CS(rr,xi6.real)
        xi8s = CS(rr,xi8.real)
        
        xi0 = xi0s(self.sbin)
        xi2 = xi2s(self.sbin)
        xi4 = xi4s(self.sbin)
        xi6 = xi6s(self.sbin)
        xi8s = xi8s(self.sbin)
            
        outc = {}
        outc["s"]=self.sbin
        outc["xi0"]=xi0
        outc["xi2"]=xi2
        outc["xi4"]=xi4
        outc["xi6"]=xi6
        outc["xi8"]=xi8
        self.outc = outc
        
        return outc    

    def get_corr(self):
        global kcorr,A11,A12,A22,A23,A33,B111,B112,B121,B122,B211,B212,B221,B222,B312,B321,B322,B422
        TNScorr.compute()
        return A11,A12,A22,A23,A33,B111,B112,B121,B122,B211,B212,B221,B222,B312,B321,B322,B422

    def read_python_input(self,datain):
        global kc,Pklin,Pdd,Pdt,Ptt
        global kcorr,Id2,Ig2,Fg2,Id2d2,Id2g2,Ig2g2,Id2theta,Ig2theta,Fg2theta
        global A11,A12,A22,A23,A33,B111,B112,B121,B122,B211,B212,B221,B222,B312,B321,B322,B422


        kc = self.carray(datain['k'])
        Pklin = self.carray(datain['Pk_lin'])
        Pdd = self.carray(datain['Pdd'])
        Pdt = self.carray(datain['Pdt'])
        Ptt = self.carray(datain['Ptt'])
        
        kcorr = self.carray(datain['kcorr'])
        Id2 = self.carray(datain['Id2'])
        Ig2 =  self.carray(datain['Ig2'])
        Fg2 =  self.carray(datain['Fg2'])
        Id2theta = self.carray(datain['Id2theta'])
        Ig2theta = self.carray(datain['Ig2theta'])
        Fg2theta = self.carray(datain['Fg2theta'])
        Id2d2 = self.carray(datain['Id2d2'])
        Id2g2 = self.carray(datain['Id2g2'])
        Ig2g2 = self.carray(datain['Ig2g2'])
        
        if 'A11' in datain:
            A11 = self.carray(datain['A11'])
            A12 = self.carray(datain['A12'])
            A22 = self.carray(datain['A22'])
            A23 = self.carray(datain['A23'])
            A33 = self.carray(datain['A33'])
            B111 = self.carray(datain['B111'])
            B112 = self.carray(datain['B112'])
            B121 = self.carray(datain['B121'])
            B122 = self.carray(datain['B122'])
            B211 = self.carray(datain['B211'])
            B212 = self.carray(datain['B212'])
            B221 = self.carray(datain['B221'])
            B222 = self.carray(datain['B222'])
            B312 = self.carray(datain['B312'])
            B321 = self.carray(datain['B321'])
            B322 = self.carray(datain['B322'])
            B422 = self.carray(datain['B422'])

        
cdef double FoG_kurtosis(double k,double mu, double sigmaV, double at) nogil:
    cdef double FoG = 1./(pow(1.+pow(k*mu*at,2),1./2.))*exp(-pow(k*mu*sigmaV,2)/(1 + pow(k*mu*at,2)));
    return FoG

cdef double FoG_Lorentzian(double k, double mu, double sigmaV) nogil:
    cdef double FoG = pow(1.+pow(k*mu*sigmaV/2,2),-2);
    return FoG

cdef double FoG(double k, double mu, double FoGdamp) nogil:
    cdef double result
    if strcmp(FoG_type, Lorentzian)==0:
        result = FoG_Lorentzian(k,mu,FoGdamp)
    else :
        result = FoG_kurtosis(k,mu,sigmaV_lin,FoGdamp)
    return result
 
cdef double Pk_gg(double k, double b1, double b2, double bg2, double bT3) nogil:
    cdef double result
    result = b1*b1*sp.Pk_dd(k) + b1*b2*sp.Pk_Id2(k) + 2*b1*bg2*sp.Pk_Ig2(k) + \
    2*b1*(bg2+ (2./5.)*bT3)*sp.Pk_Fg2(k) + \
    b2*bg2*sp.Pk_Id2g2(k) + bg2*bg2*sp.Pk_Ig2g2(k) + (1./4.)*b2*b2*sp.Pk_Id2d2(k) 
    return result
    
cdef double Pk_gt(double k,double b1, double b2, double bg2, double bT3) nogil:
    cdef double result
    result = b1*sp.Pk_dt(k) + (b2/2.)*sp.Pk_Id2theta(k) + bg2*sp.Pk_Ig2theta(k) + \
    (bg2 + (2./5.)*bT3)*sp.Pk_Fg2theta(k)
    return result

cdef double Pk_gm(double k,double b1, double b2, double bg2, double bT3) nogil:
    cdef double result
    result = b1*sp.Pk_dd(k) + (b2/2.)*sp.Pk_Id2(k) + bg2*sp.Pk_Ig2(k) + \
    (bg2 + (2./5.)*bT3)*sp.Pk_Fg2(k)
    return result


cdef double pkmu(double k,double mu, double aper, double apar, double f, double b1, double b2, double bg2, double bT3, double FoGdamp) nogil:
    cdef double Fe = apar/aper
    
    k = k/aper*pow((1 + mu*mu*(1./(Fe*Fe) - 1.)),1./2);
    mu = mu/Fe*pow((1 + mu*mu*(1./(Fe*Fe) - 1.)),-1./2);
    
    cdef double FoGx = FoG(k,mu,FoGdamp)
    cdef double TNS
    cdef double pkc
    cdef double lowk_ = 1e-5
    cdef double highk_ = 10
    cdef double damping_lowk = 1
    cdef double damping_highk = 1

    TNS = Pk_gg(k,b1,b2,bg2,bT3) + mu*mu*(2*f*Pk_gt(k,b1,b2,bg2,bT3)+ b1*b1*f*sp.cA(k,1,1) + b1*f*f*sp.cA(k,1,2) + \
     b1*b1*f*f*sp.cB(k,1,1,1) -b1*f*f*f*sp.cB(k,1,2,1) -b1*f*f*f*sp.cB(k,1,1,2) + f*f*f*f*sp.cB(k,1,2,2)) \
    + mu*mu*mu*mu*(f*f*sp.Pk_tt(k) + b1*f*f*sp.cA(k,2,2) + f*f*f*sp.cA(k,2,3) +  +b1*b1*f*f*sp.cB(k,2,1,1) \
    -b1*f*f*f*sp.cB(k,2,1,2)-b1*f*f*f*sp.cB(k,2,2,1)+f*f*f*f*sp.cB(k,2,2,2)) \
    + pow(mu,6)*(f*f*f*sp.cA(k,3,3)-b1*f*f*f*sp.cB(k,3,1,2)-b1*f*f*f*sp.cB(k,3,2,1)+f*f*f*f*sp.cB(k,3,2,2)) \
    + pow(mu,8)*f*f*f*f*sp.cB(k,4,2,2)

    if (k > highk_):
        damping_highk=exp(-pow(k/highk_ - 1,2))

    if (k < lowk_):
        damping_lowk = exp(-pow(lowk_/k - 1,2))

    pkc = TNS*FoGx*damping_highk*damping_lowk


    return pkc


cdef double pk_ell_int(double mu, void * p) nogil:
    
    cdef double k  = (<double *> p)[0]
    cdef double aper = (<double *> p)[1]
    cdef double apar = (<double *> p)[2]
    cdef double f = (<double *> p)[3]
    cdef double b1 = (<double *> p)[4]
    cdef double b2 = (<double *> p)[5]
    cdef double bg2 = (<double *> p)[6]
    cdef double bT3 = (<double *> p)[7]
    cdef double FoGdamp = (<double *> p)[8]
    cdef double l = (<double *> p)[9]
    
    cdef double result
    if (<int>l == 0):    result =(0.5/(apar*aper*aper))*pkmu(k,mu,aper,apar,f,b1,b2,bg2,bT3,FoGdamp);
    if (<int>l == 2):    result = (2.5/(apar*aper*aper))*pkmu(k,mu,aper,apar,f,b1,b2,bg2,bT3,FoGdamp)*Pl_2(mu);
    if (<int>l == 4):    result = (4.5/(apar*aper*aper))*pkmu(k,mu,aper,apar,f,b1,b2,bg2,bT3,FoGdamp)*Pl_4(mu);
    if (<int>l == 6):    result = (6.5/(apar*aper*aper))*pkmu(k,mu,aper,apar,f,b1,b2,bg2,bT3,FoGdamp)*Pl_6(mu);
    if (<int>l == 8):    result = (8.5/(apar*aper*aper))*pkmu(k,mu,aper,apar,f,b1,b2,bg2,bT3,FoGdamp)*Pl_8(mu);   
    return result

cdef double pk_ell(double k,double aper, double apar, double f, double b1, double b2, double bg2, double bT3, double FoGdamp,double l) nogil:
    cdef double * params = <double *> malloc(11*sizeof(double)) 
    cdef  double result
    if (l > lmax):
        free(params)
        result = 0
        return result
    else :
        params[0] = k
        params[1] = aper
        params[2] = apar
        params[3] = f
        params[4] = b1
        params[5] = b2
        params[6] = bg2
        params[7] = bT3
        params[8] = FoGdamp
        params[9] = l
        result = int_gsl_qag(pk_ell_int, params,-1.,1.)
        free(params)
        return result

cdef void TNS_multipoles(double aper, double apar, double f, double b1, double b2, double bg2, double bT3, double FoGdamp) nogil:
    global p0,p2,p4,p6,p8,calc
    cdef Py_ssize_t i
    calc =+ 1
    for i in prange(0,Npo,nogil=True):
        p0[i] = pk_ell(kc[i],aper,apar,f,b1,b2,bg2,bT3,FoGdamp,0)
        p2[i] = pk_ell(kc[i],aper,apar,f,b1,b2,bg2,bT3,FoGdamp,2)
        p4[i] = pk_ell(kc[i],aper,apar,f,b1,b2,bg2,bT3,FoGdamp,4)
        p6[i] = pk_ell(kc[i],aper,apar,f,b1,b2,bg2,bT3,FoGdamp,6)
        p8[i] = pk_ell(kc[i],aper,apar,f,b1,b2,bg2,bT3,FoGdamp,8)
        
        

        

    
    
        

    
        
        

