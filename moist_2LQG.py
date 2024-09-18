import numpy as np
import matplotlib.pylab as plt
from random import random
from numba import jit
import scipy.fft as sci_fft
import time as ti
import xarray as xr

class MoistQGModel:

    def __init__(self, opt = "full", moist = "moist", N=128, N2=128, Lx=72., Ly=96., nu=1e-6, tau_d=100., 
                 tau_f=15., beta=0.2, sigma=3.5, U_1=1., C=2., L=0.2, 
                 Er=.1, g=0.04, init="cold", filename = "moist_QG_run1.nc",
                 res_filename="res_file", nworker=4, tot_time=11000, 
                 dt=0.025, lim=1000, st=1):
      
        # Initialize model parameters as instance attributes
        self.opt = opt # linear, nonlinear or full
        self.moist = "moist" #moist or dry

        self.filename = filename
        self.res_filename = res_filename 

        self.tot_time = tot_time #Length of run (model time-units)
        self.dt = dt #Timestep
        self.ts = int( float(tot_time) / dt ) #Total timesteps
        self.g = g #leapfrog filter coefficient
        self.lim = lim #Start saving after this time (model time-units)
        self.st = st #How often to record data (model time-units)

        self.init = init

        self.N = N #zonal size of spectral decomposition
        self.N2 = N2 #meridional size of spectral decomposition
        self.Lx = Lx #size of x
        self.Ly = Ly #size of y

        self.fft = sci_fft
        self.nworker = nworker

        #dry model parameters
        self.tau_d = tau_d #Newtonian relaxation time-scale for interface
        self.tau_f = tau_f #surface friction 
        self.beta = beta #beta
        self.sigma = sigma #width of jet
        self.U_1 = U_1 #wind strength
        self.nu = nu #viscous dissipation

        #moist parameters
        self.C = C #linearized Clausius-Clapeyron parameter
        self.L = L #non-dimensional measure of the strength of latent heating
        self.Er = Er #Evaporation rate

        # Initialize arrays as instance attributes
        self.x = np.linspace(-self.Lx / 2, self.Lx / 2, self.N, endpoint=False)
        self.y = np.linspace(-self.Ly / 2, self.Ly / 2, self.N2, endpoint=False)
        self.kk = np.fft.rfftfreq(self.N, self.Lx / float(self.N) / 2. / np.pi) #zonal wavenumbers
        self.ll = np.fft.fftfreq(self.N2, self.Ly / float(self.N2) / 2. / np.pi) #meridional wavenumbers
        self.Lapl = -(np.expand_dims(self.ll, 1) ** 2 + np.expand_dims(self.kk, 0) ** 2)

        #Initialize spectral arrays, only need 3 time-steps
        self.psic_1 = np.zeros((3, self.N2, int(self.N / 2 + 1))).astype(complex)
        self.psic_2 = np.zeros((3, self.N2, int(self.N / 2 + 1))).astype(complex)
        self.qc_1 = np.zeros((3, self.N2, int(self.N / 2 + 1))).astype(complex)
        self.qc_2 = np.zeros((3, self.N2, int(self.N / 2 + 1))).astype(complex)
        self.vorc_1 = np.zeros((3, self.N2, int(self.N / 2 + 1))).astype(complex)
        self.vorc_2 = np.zeros((3, self.N2, int(self.N / 2 + 1))).astype(complex)

        self.time = np.zeros((self.tot_time - self.lim) // st)

        self.mc = np.zeros((3, self.N2, self.N // 2 + 1)).astype(complex) #moisture

        #Initialize physical arrays, only need 3 time-steps
        self.psi_1 = np.zeros((3, self.N2, self.N))
        self.psi_2 = np.zeros((3, self.N2, self.N))
        self.q_1 = np.zeros((3, self.N2, self.N))
        self.q_2 = np.zeros((3, self.N2, self.N))

        self.ds = xr.Dataset({
    'beta' : self.beta, 
    'tau_f' : self.tau_f, 
    'tau_d' : self.tau_d, 
    'U_1' : self.U_1, 
    'sigma' : self.sigma, 
    'beta' : self.beta, 
    'L' : self.L, 
    'C' : self.C, 
    'E' : self.Er,
    'u1': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'u2': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'tau': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'm': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'P': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'E': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'w': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'wskew': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'eke1': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'eke2': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'emf1': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'emf2': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'ehf1': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'ehf2': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'epv1': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'epv2': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'empv': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'eape': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'eape_eke': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),
    'P_eddy': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),    
    'sat_frac': (('time', 'y'), np.zeros(((tot_time - lim) // st, N2))),    
}, coords={
    'time': np.zeros((tot_time - lim) // st),
    'y': self.y
})

    #######################################################
    #  Spectral functions

    #@jit(nopython=True)
    def ptq(self, kk, ll, ps1, ps2):
        """
        Calculate PV
        in:
        meridional wavemnumber ll, zonal wavenumber kk, psi1(l,k), psi2(l,k)
        """
        q1 = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 ) * ps1 - (ps1 - ps2) # -(k^2 + l^2) * psi_1 -0.5*(psi_1-psi_2)
        q2 = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 ) * ps2 + (ps1 - ps2) # -(k^2 + l^2) * psi_2 +0.5*(psi_1-psi_2)
 
        return q1, q2


    #@jit(nopython=True)
    def qtp(self, kk, ll, q1_s, q2_s):
        """Invert PV"""
        divider =  ( np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)  # (psi_1 + psi_2)/2
        divider[0, 0] = np.nan
        psi_bt = -(q1_s + q2_s) / divider / 2.0  # (psi_1 + psi_2)/2
        psi_bt[0, 0] = 0.

        psi_bc = -(q1_s - q2_s) / (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2

        psi1 = psi_bt + psi_bc
        psi2 = psi_bt - psi_bc

        return psi1, psi2

    #@jit(nopython=True)
    def qtp_3d(self, kk, ll, q1_s, q2_s):
        """Invert PV"""

        divider =  ( np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)  # (psi_1 + psi_2)/2
        divider[0, 0] = np.nan
        psi_bt = -(q1_s + q2_s) / divider  /2.0 # (psi_1 + psi_2)/2
        psi_bt[:, 0, 0] = 0.

        psi_bc = -(q1_s - q2_s) / (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2

        psi1 = psi_bt + psi_bc
        psi2 = psi_bt - psi_bc

        return psi1, psi2

    #@jit(nopython=True)
    def grad( self, field, kk, ll ):

        d1, d2 = np.shape( field )
        grad = np.zeros( ( d1, d2 ) ) + 0.j
        grad[:, :] = 1.j * np.expand_dims(ll, 1) + 1.j * np.expand_dims(kk, 0)

        return grad * field

    #@jit(nopython=True)
    def Laplace( self, field, kk, ll ):

        d1, d2 = np.shape( field )
        Lapl = np.zeros( ( d1, d2 ) )
        Lapl[:, :] = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)


        return Lapl * field

    #@jit(nopython=True, parallel = True)
    def exponential_cutoff( self, data, a, s, kcut ):
          d1, d2 = np.shape(data)
          i, j = np.indices(data.shape)  # Create index arrays

          # Calculate exponential factors using vectorized operations
          F_i = np.where((i > 9) & (i <= d1 / 2), 
                        np.exp(-a * ((i.astype(float) - kcut) / (d1 / 2 - 1 - kcut)) ** s), 1)
          F_i = np.where((i > d1 / 2) & (i < (d1 - 10)), 
                        np.exp(-a * (((d1 - i).astype(float) - kcut) / (d1 / 2 - 1 - kcut)) ** s), F_i)
          F_j = np.where(j > 9, 
                        np.exp(-a * ((j.astype(float) - kcut) / (d2 - 1 - kcut)) ** s), 1)

          # Multiply the factors to get the final filter
          F = F_i * F_j
          return F

    #######################################################
    #  Time-stepping functions
    def initialize( self):
        # Initialize equilibrium interface height and sponge
        self.sponge = np.zeros(self.N2)
        self.u_eq = np.zeros((self.N2, self.N))
        for i in range( self.N2 ):
          y1 = float( i - self.N2 /2) * (self.y[1] - self.y[0] )
          y2 = float(min(i, self.N2 -i - 1)) * (self.y[1] - self.y[0] )
          self.sponge[i] = self.U_1 / (np.cosh(abs(y2/self.sigma)))**2 
          self.u_eq[i, :] = self.U_1 * ( 1. / (np.cosh(abs(y1/self.sigma)))**2 - 1. / (np.cosh(abs(y2/self.sigma)))**2  )

        self.psi_Rc = -np.fft.rfft2(  self.u_eq ) / 1.j / self.ll[:, np.newaxis]
        self.psi_Rc[0] = 0.
        self.psi_R = np.fft.irfft2( self.psi_Rc )

        # Set up initial conditions
        if self.init == "cold":
            self.psic_1[0] = [ [ random() for i in range(self.N // 2 + 1 ) ] for j in range(self.N2) ]
            self.psic_2[0] = [ [ random() for i in range(self.N // 2 + 1 ) ] for j in range(self.N2) ]

            # Transfer values:
            self.psic_1[1, :, :] = self.psic_1[0, :, :]
            self.psic_2[1, :, :] = self.psic_2[0, :, :]

            # Calculate initial PV
            for i in range(2):
                self.vorc_1[i], self.vorc_2[i] = self.ptq(self.kk, self.ll, self.psic_1[i], self.psic_2[i])
                self.q_1[i] = self.fft.irfft2(self.vorc_1[i], workers=self.nworker) + self.beta * self.y[:, np.newaxis]
                self.q_2[i] = self.fft.irfft2(self.vorc_2[i], workers=self.nworker) + self.beta * self.y[:, np.newaxis]
                self.qc_1[i] = self.fft.rfft2(self.q_1[i], workers=self.nworker )
                self.qc_2[i] = self.fft.rfft2(self.q_2[i], workers=self.nworker )
                self.psi_1[1] = self.fft.irfft2(self.psic_1[1], workers=self.nworker )
                self.psi_2[1] = self.fft.irfft2(self.psic_2[1], workers=self.nworker )

            # Start moisture at 50% saturation
            m = self.C * (self.psi_1[1] - self.psi_2[1]) / 2.

            self.mc[0] = self.fft.rfft2(m.real, workers=self.nworker )
            self.mc[1] = self.fft.rfft2(m.real, workers=self.nworker )

            self.t0 = 1
                      
        elif self.init == "load":
            dsr = xr.open_dataset(self.res_filename)
            self.psic_1 = dsr["psic_1"]
            self.psic_2 = dsr["psic_2"]
            self.qc_1 = dsr["qc_1"]
            self.qc_2 = dsr["qc_2"]
            self.mc = dsr["mc"]
            self.t0 = dsr["t0"]

            for i in range(2):
                self.vorc_1[i], self.vorc_2[i] = self.ptq(self.kk, self.ll, self.psic_1[i], self.psic_2[i])
                self.q_1[i] = self.fft.irfft2(self.vorc_1[i], workers=self.nworker) + self.beta * self.y[:, np.newaxis]
                self.q_2[i] = self.fft.irfft2(self.vorc_2[i], workers=self.nworker) + self.beta * self.y[:, np.newaxis]
                self.psi_1[i] = self.fft.irfft2(self.psic_1[i], workers=self.nworker)
                self.psi_2[i] = self.fft.irfft2(self.psic_2[i], workers=self.nworker)

        return 0



    def calc_nl( self, psi, qc ):
      """"Calculate non-linear terms, with Orszag 3/2 de-aliasing"""

      N2, N = np.shape( psi )
      ex = int(N *  3 / 2)# - 1
      ex2 = int(N2 * 3 / 2)# - 1
      temp1 = np.zeros( ( ex2, ex ) ).astype( complex )
      temp2 = np.zeros( ( ex2, ex ) ).astype( complex )
      temp4 = np.zeros( ( N2, N ) ).astype( complex )	#Final array

      #Pad values:
      temp1[:N2//2, :N] = psi[:N2//2, :N]
      temp1[ex2-N2//2:, :N] = psi[N2//2:, :N]

      temp2[:N2//2, :N] = qc[:N2//2, :N]
      temp2[ex2-N2//2:, :N] = qc[N2//2:, :N]

      #Fourier transform product, normalize, and filter:
      temp3 = np.fft.rfft2( np.fft.irfft2( temp1 ) * np.fft.irfft2( temp2 ) ) * 9. / 4.
      temp4[:N2//2, :N] = temp3[:N2//2, :N]
      temp4[N2//2:, :N] = temp3[ex2-N2//2:, :N]

      return temp4

    #@jit(nopython=True)
    def jacobian_prep(self, kk, ll, psi, qc):
        #kk, ll, psi, qc = kk, ll, psic_1[1, :, :], vorc_1[1, :, :]
        kk2, ll2 =np.expand_dims(kk, 0), np.expand_dims(ll, 1)
        dpsi_dx = 1.j * kk2 * psi
        dpsi_dy = 1.j * ll2 * psi

        dq_dx = 1.j * kk2 * qc
        dq_dy = 1.j * ll2 * qc
        return dpsi_dx, dq_dy, dpsi_dy ,dq_dx

    def nlterm(self, kk, ll, psi, qc):
          """"Calculate Jacobian"""

          dpsi_dx, dq_dy, dpsi_dy, dq_dx = self.jacobian_prep(kk, ll, psi, qc) 

          return  self.calc_nl(dpsi_dx, dq_dy ) - self.calc_nl( dpsi_dy, dq_dx )

    def lterm(self, layer, kk, ll, psi1, psi2, psic2, q):
          """"Calculate linear term"""
          #Relax interface
          forc = pow(-1., layer+1.)*(psi1 - psi2 - self.psi_R) / self.tau_d 
  
          #Sponge
          forc -= self.sponge[:, np.newaxis] * (q - np.mean( q, axis = 1)[:, np.newaxis] )
            
          if layer == 1:
              return self.fft.rfft2( forc, workers = self.nworker)
          else:
              return self.fft.rfft2( forc, workers = self.nworker) - self.Laplace( psic2, kk, ll ) / self.tau_f

    def calc_precip_evap(self, kk, ll, mc, psic1, psic2, F):
          """"Calculate linear term"""

          #Convert to real space
          m = self.fft.irfft2(mc, workers = self.nworker)
          psi1 = self.fft.irfft2( psic1, workers = self.nworker)
          psi2 = self.fft.irfft2( psic2, workers = self.nworker)
          u2 = self.fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic1, workers = self.nworker)
          v2 = self.fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic2, workers = self.nworker)

          #Calculate precip
          sat_def = (self.C * (psi1[:,:] - psi2[:, :]) ) - m

          sat_def_mask=sat_def < 0.

          P = np.where(sat_def_mask, -sat_def , 0)
          E = np.where(~sat_def_mask, self.Er * np.sqrt(u2 ** 2 + v2 ** 2) * sat_def, 0)

          Pc = self.fft.rfft2( P, workers = self.nworker) * F #F is exponential cut-off filter
          Ec = self.fft.rfft2( E, workers = self.nworker) * F #F is exponential cut-off filter

          return m, Pc, Ec
    
    def calc_omega(self, kk, ll, psic1, psic2, Pc):
        vort1 = self.Laplace( psic1, kk, ll )
        term1 = self.nlterm( kk, ll, psic1, vort1)

        vort2 = self.Laplace( psic2, kk, ll )
        term2 = self.nlterm( kk, ll, psic2, vort2)

        tauc = psic1 - psic2
        term3 = self.Laplace( self.nlterm( kk, ll, psic2, tauc), kk, ll )

        v1 = 1.j * np.expand_dims(kk, 0) * psic1[:, :]
        v2 = 1.j * np.expand_dims(kk, 0) * psic2[:, :]
        term4 = self.beta * (v1 - v2) 

        term5 = self.Laplace( self.L * Pc, kk, ll )

        term6 = self.Laplace( tauc - self.psi_Rc, kk, ll ) / self.tau_d

        div_ageo = term1 - term2 - term3 + term4 + term5 - term6
        return div_ageo / (np.expand_dims(self.ll, 1) ** 2 + np.expand_dims(self.kk, 0) ** 2 + 2)

    #@jit(nopython=True)
    def fs(self, ovar, rhs, det, nu, kk, ll):
          """Forward Step: q^t-1 / ( 1 + 2. dt * nu * (k^4 + l^4 ) ) + RHS"""
          mult = det / ( 1. + det * nu * (np.expand_dims(kk, 0) ** 4 + np.expand_dims(ll, 1) ** 4) )
          return mult * (ovar / det + rhs)

    #@jit(nopython=True)
    def lf(self, oovar, rhs, det, nu, kk, ll):
          """Leap frog timestepping: q^t-2 / ( 1 + 2. * dt * nu * (k^4 + l^4 ) ) + RHS"""
          mult = 2. * det / ( 1. + 2. * det * nu * (np.expand_dims(kk, 0) ** 4 + np.expand_dims(ll, 1) ** 4) )
          return mult * (oovar / det / 2. + rhs)

    #@jit(nopython=True)
    def filt(self, var, ovar, nvar, g):
        """Leapfrog filtering"""
        return var + g * (ovar - 2. * var + nvar )
    
    def write_restart(self, time_index):
        res = xr.Dataset({
    'beta' : self.beta, 
    'tau_f' : self.tau_f, 
    'tau_d' : self.tau_d, 
    'U_1' : self.U_1, 
    'sigma' : self.sigma, 
    'L' : self.L, 
    'C' : self.C, 
    'E' : self.Er,
    'time_index' : time_index,
    'psic1': (('time', 'y', 'x'), self.psic_1),
    'psic2': (('time', 'y', 'x'), self.psic_2),
    'qc1': (('time', 'y', 'x'), self.qc_1),
    'qc2': (('time', 'y', 'x'), self.qc_2),
    'mc': (('time', 'y'), self.mc),  
}, coords={
    'time': np.arange(0, 3, 1),
    'y': self.y,
    'x': self.x
})
        res.to_netcdf(path=self.res_filename)

    def run_simulation(self):
        """Run the simulation"""

        self.initialize()

        rhs1 = np.zeros((self.N2, self.N // 2 + 1)).astype(complex)
        rhs2 = np.zeros_like(rhs1).astype(complex)
        mrhs = np.zeros_like(rhs1).astype(complex)

        F = self.exponential_cutoff( self.psic_1[0], np.log(1. + 400. * np.pi / float(self.N) ), 6, 7 )

        #Timestepping:
        for i in range( self.t0, self.ts ):
          start = ti.time()
          if i % 1000 == 0:
              print("Timestep:", i, " / ", self.ts)

          if self.opt != "linear":
              #NL terms -J(psi, qc) - beta * v (for some reason its more stable to do beta v separately)
              rhs1[:, :] = -self.nlterm( self.kk, self.ll, self.psic_1[1, :, :], self.vorc_1[1, :, :]) - self.beta * 1.j * self.kk[np.newaxis, :] * self.psic_1[1, :, :]
              rhs2[:, :] = -self.nlterm( self.kk, self.ll, self.psic_2[1, :, :], self.vorc_2[1, :, :]) - self.beta * 1.j * self.kk[np.newaxis, :] * self.psic_2[1, :, :]
          if self.opt != "nonlinear":
              #Linear terms
              rhs1 += self.lterm(1, self.kk, self.ll, self.psi_1[1], self.psi_2[1], self.psic_2[1], self.q_1[1])
              rhs2 += self.lterm(2, self.kk, self.ll, self.psi_1[1], self.psi_2[1], self.psic_2[1], self.q_2[1],)

          if i == 1 and self.init == 'cold':
              #Forward step
              self.qc_1[2, :] = self.fs(self.qc_1[1, :, :], rhs1[:], self.dt, self.nu, self.kk, self.ll)
              self.qc_2[2, :] = self.fs(self.qc_2[1, :, :], rhs2[:], self.dt, self.nu, self.kk, self.ll)
          else:
              #Leapfrog step
              self.qc_1[2, :] = self.lf(self.qc_1[0, :, :], rhs1[:], self.dt, self.nu, self.kk, self.ll)
              self.qc_2[2, :] = self.lf(self.qc_2[0, :, :], rhs2[:], self.dt, self.nu, self.kk, self.ll)


          if self.moist == "moist":
            if self.opt != "linear":
                  #NL terms -J(psi, m)
                  mrhs = -self.nlterm( self.kk, self.ll, self.psic_2[1, :, :], self.mc[1, :, :])
              
                  #need to time-step to diagnose precip
                  if i == 1 and self.init == 'cold':
                  #Forward step
                        self.mc[2, :] = self.fs(self.mc[1, :, :], mrhs[:], self.dt, self.nu, self.kk, self.ll)
                  else:
                  #Leapfrog step
                        self.mc[2, :] = self.lf(self.mc[0, :, :], mrhs[:], self.dt, self.nu, self.kk, self.ll)

            #############################################
            #Calculate precip, then adjust fields
            m, Pc, Ec = self.calc_precip_evap(self.kk, self.ll, self.mc[2], self.psic_1[1], self.psic_2[1], F)

            #Adjust fields, time-step
            rhs1 -= self.L * Pc
            rhs2 += self.L * Pc
            mrhs -= Pc - Ec
        
            if i == 1 and self.init == 'cold':
                #Forward step
                self.qc_1[2, :] = self.fs(self.qc_1[1, :, :], rhs1[:], self.dt, self.nu, self.kk, self.ll)
                self.qc_2[2, :] = self.fs(self.qc_2[1, :, :], rhs2[:], self.dt, self.nu, self.kk, self.ll)
            else:
                #Leapfrog step
                self.qc_1[2, :] = self.lf(self.qc_1[0, :, :], rhs1[:], self.dt, self.nu, self.kk, self.ll)
                self.qc_2[2, :] = self.lf(self.qc_2[0, :, :], rhs2[:], self.dt, self.nu, self.kk, self.ll)

            #############################################
            #Final adjustments of m: omega equation and sponge
            cdiv_ageo = self.calc_omega(self.kk, self.ll, self.psic_1[1], self.psic_2[1], Pc) #need this divergence later
            mrhs -= cdiv_ageo
            #sponge
            mrhs -= self.fft.rfft2(self.sponge[:, np.newaxis] * (m - np.mean( m, axis = 1)[:, np.newaxis] ), workers = self.nworker)

            #x 3ms
            #timeit
            if i == 1 and self.init == 'cold':
                #Forward step
                self.mc[2, :] = self.fs(self.mc[1, :, :], mrhs[:], self.dt, self.nu, self.kk, self.ll)
            else:
                #Leapfrog step
                self.mc[2, :] = self.fs(self.mc[0, :, :], mrhs[:], self.dt, self.nu, self.kk, self.ll)

          if i > 1:
          #Leapfrog filter
              self.qc_1[1, :] = self.filt( self.qc_1[1, :], self.qc_1[0, :], self.qc_1[2, :], self.g)
              self.qc_2[1, :] = self.filt( self.qc_2[1, :], self.qc_2[0, :], self.qc_2[2, :], self.g)
              self.mc[1, :] = self.filt( self.mc[1, :], self.mc[0, :], self.mc[2, :], self.g)

          self.q_1[0:2] = self.fft.irfft2( self.qc_1[1:], workers = self.nworker)
          self.q_2[0:2] = self.fft.irfft2( self.qc_2[1:], workers = self.nworker)

          #Subtract off beta and invert
          self.vorc_1[0:2] = self.fft.rfft2( self.q_1[0:2] - self.beta * self.y[:, np.newaxis], workers = self.nworker)
          self.vorc_2[0:2] = self.fft.rfft2( self.q_2[0:2] - self.beta * self.y[:, np.newaxis], workers = self.nworker)
          self.psic_1[0:2], self.psic_2[0:2] = self.qtp_3d( self.kk, self.ll, self.vorc_1[0:2], self.vorc_2[0:2] )

          self.psi_1[0:2] = self.fft.irfft2( self.psic_1[0:2] , workers = self.nworker)
          self.psi_2[0:2] = self.fft.irfft2( self.psic_2[0:2] , workers = self.nworker)

          #Transfer values:
          self.qc_1[0:2, :, :] = self.qc_1[1:, :, :]
          self.qc_2[0:2, :, :] = self.qc_2[1:, :, :]
          self.mc[0:2, :, :] = self.mc[1:, :, :]

          if int(float(i) * self.dt) > self.lim: 
              if i % int(float(self.st) / self.dt) == 0: 
                  time_index = (int(i * self.dt) - self.lim) // self.st
                  self.time[time_index] = float(i) * self.dt

                  u1 = self.fft.irfft2( -1.j * self.ll[:, np.newaxis] * self.psic_1[1], workers = self.nworker)
                  u2 = self.fft.irfft2( -1.j * self.ll[:, np.newaxis] * self.psic_2[1], workers = self.nworker)
                  v1 = self.fft.irfft2( 1.j * self.kk[np.newaxis, :] * self.psic_1[1], workers = self.nworker)
                  v2 = self.fft.irfft2( 1.j * self.kk[np.newaxis, :] * self.psic_2[1], workers = self.nworker)
                  tau = self.fft.irfft2( self.psic_1[1] - self.psic_2[1], workers = self.nworker)

                  zmu1 = np.mean( u1, axis = 1 )
                  zmu2 = np.mean( u2, axis = 1 )
                  zmtau = np.mean( tau, axis = 1 )

                  eddy_u1 = u1 - zmu1[:, np.newaxis]
                  eddy_u2 = u2 - zmu2[:, np.newaxis]
                  eddy_v1 = v1 - np.mean( v1, axis = 1)[:, np.newaxis]
                  eddy_v2 = v2 - np.mean( v2, axis = 1)[:, np.newaxis]
                  eddy_tau = tau - zmtau[:, np.newaxis]

                  self.ds['u1'][time_index, :] = zmu1[:]
                  self.ds['u2'][time_index, :] = zmu2[:]
                  self.ds['tau'][time_index, :] = zmtau[:]

                  zeke1 = np.mean( eddy_u1 ** 2 + eddy_v1 ** 2, axis = 1) / 2.
                  zeke2 = np.mean( eddy_u2 ** 2 + eddy_v1 ** 2, axis = 1) / 2.
                  zemf1 = np.mean( eddy_u1 * eddy_v1, axis = 1)
                  zemf2 = np.mean( eddy_u2 * eddy_v2, axis = 1)
                  zehf1 = np.mean( eddy_v1 * eddy_tau, axis = 1)
                  zehf2 = np.mean( eddy_v2 * eddy_tau, axis = 1)
                  epv1 = zehf1 - np.gradient(zemf1, self.y)
                  epv2 = zehf2 - np.gradient(zemf2, self.y)

                  self.ds['eke1'][time_index, :] = zeke1
                  self.ds['eke2'][time_index, :] = zeke2
                  self.ds['emf1'][time_index, :] = zemf1 
                  self.ds['emf2'][time_index, :] = zemf2
                  self.ds['ehf1'][time_index, :] = zehf1
                  self.ds['ehf2'][time_index, :] = zehf2
                  self.ds['epv1'][time_index, :] = epv1
                  self.ds['epv2'][time_index, :] = epv2

                  self.ds['eape'][time_index, :] = np.mean( eddy_tau * eddy_tau, axis = 1)	
                  w = self.fft.irfft2( cdiv_ageo )
                  zw = np.mean(w, axis = 1)
                  eddy_w = w - zw[:, np.newaxis]
                  self.ds['eape_eke'][time_index, :] = np.mean( eddy_tau * eddy_w, axis = 1)	

                  if self.moist == "moist":
                      m = self.fft.irfft2( self.mc[1], workers = self.nworker)
                      P = self.fft.irfft2( Pc, workers = self.nworker)
                      E = self.fft.irfft2( Ec, workers = self.nworker)

                      sat_def = (self.C * tau ) - m
                      sat_def_mask = sat_def < 0.
                      sat_def_mask = np.sum(sat_def_mask, axis = 1)
                      self.ds['sat_frac'][time_index, :] = sat_def_mask / float(self.N)

                      zm = np.mean(m, axis = 1)
                      zP = np.mean(P, axis = 1)
                      zE = np.mean(E, axis = 1)

                      self.ds['m'][time_index, :] = zm
                      self.ds['P'][time_index, :] = zP
                      self.ds['E'][time_index, :] = zE
                      self.ds['w'][time_index, :] = zw

                      self.ds['wskew'][time_index, :] = -np.mean( eddy_w ** 3, axis = 1 ) / np.mean( eddy_w ** 2, axis = 1 ) ** (3. / 2.)
                      eddy_P = P[:, :] - zP[:, np.newaxis]
                      self.ds['P_eddy'][time_index, :] = np.mean( eddy_tau * eddy_P, axis = 1)

                      eddy_m = m - zm[:, np.newaxis]
                      zem = np.mean( eddy_v2 * eddy_m, axis = 1)
                      self.ds['empv'][time_index, :] = zehf2/(1. - self.L) - np.gradient(zemf2, self.y) + self.L/(1. - self.L) * zem 

                  if int(float(i) * self.dt) % 100 == 0:
                    print("saving data and restart file")
                    self.ds.to_netcdf(path=self.filename)
                    #self.write_restart(time_index)

          end = ti.time()
          if i % 1000 == 0:
              delt = (end - start)
              time_left = delt * (float(self.ts) - float(i))
              print("1 iteration = %s" % delt)
              print("Estimated time left: %0.1f" % time_left)


if __name__ == '__main__':
  model = MoistQGModel(tot_time=2000)
  model.run_simulation()