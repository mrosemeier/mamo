import numpy as np
from pasta.plate import Plate


# micro properties as of Krimmer PhD
'''f
epoxy_resin = Material()
epoxy_resin.set_props_iso(E1=3.218E+09,
                          nu12=0.37,
                          rho=1.15E+03)
epoxy_resin.G23

gfecr = Material()
gfecr.set_props(E1=77.314E+09,
                E2=81.383E+09,
                E3=81.383E+09,
                nu12=0.21,
                nu13=0.21,
                nu23=0.22,
                G12=33.326E+09,
                G13=33.326E+09,
                G23=33.326E+09,
                rho=2.6E+03)

cfecr = Material()
cfecr.set_props(E1=218.620E+09,
                E2=14.137E+09,
                E3=14.137E+09,
                nu12=0.33,
                nu13=0.33,
                nu23=0.49,
                G12=71.528E+09,
                G13=71.528E+09,
                G23=5.315E+09,
                rho=999999E+03)
'''


class CompositeCFMh(object):
    ''' Implementation of the CFM_h (german ZFM_h) model as of Krimmer 2016
    '''

    def __init__(self, fiber, matrix, fu):
        self.f = fiber
        self.m = matrix

        # ondulation correction for stiffness and poisson's ratio
        # according to Krimmer 2014, eq. 3.1
        self.f.E1 = fu * self.f.E1
        self.f.nu12 = self.f.nu13 = fu * self.f.nu12

    def init_fvf_measurements(self,
                              mA_tot,
                              mA_F,
                              mA_T,
                              psi_M,
                              rho_Gl,
                              rho_T,
                              thickness,
                              rho_R,
                              fkor,
                              rho_S,
                              psi_SF,
                              ):
        ''' determine fvf from material measurements
        '''
        self.mA_tot = mA_tot
        self.mA_F = mA_F
        self.mA_T = mA_T
        #self.psi_M = psi_M
        self.rho_Gl = rho_Gl
        self.rho_T = rho_T
        self.rho_R = rho_R
        self.thickness = thickness
        self.fkor = fkor
        self.rho_S = rho_S
        self.psi_SF = psi_SF

        # self.psi_Gl = mA_F / mA_tot  # mass fraction fiber
        self.psi_F = mA_F / mA_tot  # mass fraction fiber+sizing

        # self.psi_S = self.psi_Gl * (psi_SF / (1 - psi_SF))  # eq.3.39
        self.psi_S = psi_SF * self.psi_F
        self.psi_Gl = (1 - psi_SF) * self.psi_F

        self.psi_T = mA_T / mA_tot  # mass fraction stitching thread

        psi_M = 1 - self.psi_Gl - self.psi_S - self.psi_T  # psi_M determ_tined
        #self.psi_T = 1 - self.psi_Gl - psi_M

        # eq.3.41 # mass fraction fabric
        self.psi_f = self.psi_Gl + self.psi_S + self.psi_T

        self.mA_S = self.psi_S * mA_tot
        self.mA_Gl = self.psi_Gl * mA_tot

        # self.mA_G = mA_F + mA_T + self.mA_S  # grammage fabric
        self.mA_G = mA_F + mA_T

        self.mA_M = mA_tot - self.mA_G  # grammage matrix

        self.rho_G = self.psi_f / \
            (self.psi_Gl / rho_Gl + self.psi_S / rho_S +
             self.psi_T / rho_T)  # eq.3.44 density fabric

        self.rho_F = self.psi_F / \
            (self.psi_Gl / rho_Gl + self.psi_S /
             rho_S)  # density fabric plus sizing

        # self.rho_tot = mA_tot / thickness  # total measured density

        self.rho_tot = 1 / \
            (self.psi_Gl / rho_Gl + self.psi_S / rho_S + self.psi_T /
             rho_T + psi_M / rho_R)  # eq. 3.43

        self.rho_meas = self.rho_tot / self.fkor  # eq. 3.45

        # matrix density
        # eq. 3.47
        self.rho_M = self.rho_meas * psi_M / \
            (1 - self.rho_meas *
             (self.psi_Gl / rho_Gl + self.psi_S / rho_S + self.psi_T / rho_T))

        self.xi_M = rho_R / self.rho_M - 1

        '''
        self.fvf = self.psi_f * rho_R / \
            (self.psi_f * rho_R + (1 - self.psi_f) * rho_R)  # eq.3.49
        '''
        self.fvf = self.mA_G * fkor / thickness / \
            self.rho_G  # fiber volume fraction eq. 3.51

        # self.fvf_M = self.psi_f * self.rho_M / \
        #    (self.psi_f * self.rho_M + (1 - self.psi_f) * self.rho_M)

        # correct matrix density
        #self.m.rho = self.rho_M

        # layer thicknesses
        #self.t_Gl = self.mA_Gl / (rho_Gl * self.fvf)
        self.t_f = self.mA_G / (self.rho_G * self.fvf)
        self.t_T = mA_T / (self.rho_T * self.fvf)

    def lamina_properties(self, fvf):
        ''' use either a random vector for fvf or self.fvf
        '''
        self.fvf = fvf

        sqrt3 = np.sqrt(3.)
        twosqrt3fvfpi = 2. * sqrt3 * fvf / np.pi
        self.twosqrt3fvfpi = twosqrt3fvfpi
        oneminemeft = 1. - self.m.E2 / self.f.E2

        # eq 5
        def eq_five(first_term_t, oneminfrac):
            result = 2. * first_term_t / sqrt3 * (sqrt3 / 2. - np.sqrt(twosqrt3fvfpi) - np.pi / (2. * oneminfrac) + ((np.pi / 2. + np.arctan(
                (np.sqrt(twosqrt3fvfpi) * oneminfrac) / np.sqrt(1. - twosqrt3fvfpi * oneminfrac**2))) / (oneminfrac * np.sqrt(1. - twosqrt3fvfpi * oneminfrac**2))))
            return result

        self.ET = eq_five(self.m.E1, oneminemeft)

        # eq. 6
        self.nuMTprime = (1. - (self.f.nu12 + self.f.nu23) * self.m.E1 /
                          (self.m.nu23 * (self.f.E1 + self.f.E2))) * self.m.nu23

        A = self.ET / self.m.E2 * (1. - self.nuMTprime**2) + self.nuMTprime**2

        self.EMTprime = (self.f.E1 * A * fvf + self.ET * (1. - fvf)) /\
            (self.f.E1 / self.m.E1 * (self.ET / self.m.E2 * (1. - 3. * self.nuMTprime**2 - 2. * self.nuMTprime**3) +
                                      2. * (self.nuMTprime**2 + self.nuMTprime**3)) * fvf +
             A * (1. - fvf))

        # eq. 7
        self.nuMLprime = (1. - (self.f.nu12 * self.m.E1) /
                          (self.m.nu12 * self.f.E2)) * self.m.nu12

        self.EMLprime = (self.ET * (1. - self.nuMLprime) + self.m.E1 * self.nuMLprime) /\
            (self.ET / self.m.E1 * (1. - self.nuMLprime) + self.nuMLprime +
             2. * self.nuMLprime**2 * (1. - self.ET / self.m.E1))

        # eq 8
        self.E1 = self.ELprime = self.f.E1 * \
            fvf + self.EMLprime * (1. - fvf)

        oneminemtprimeeft = 1. - self.EMTprime / self.f.E2
        # eq 9
        self.E2 = self.E3 = self.ETprime = eq_five(
            self.EMTprime, oneminemtprimeeft)

        # eq 10
        self.GMLTprime = (self.EMTprime + self.EMLprime) /\
            (4. * (1. + self.m.nu12))

        self.nuMTTprime = self.m.nu23 * (1 + fvf * (self.f.E1 / self.m.E1 * (1 + self.nuMTprime) - 1.)) / \
            (1. + fvf * (self.f.E1 / self.m.E1 *
                         (1. + self.nuMTprime**2 * (self.m.E1 / self.ET - 1.)) - 1.))

        self.GMTTprime = self.EMTprime / (2. * (1. + self.nuMTTprime))

        # eq. 11
        onemingmltprimegflt = 1 - self.GMLTprime / self.f.G12
        self.G12 = self.G13 = self.GLTprime = eq_five(
            self.GMLTprime, onemingmltprimegflt)

        # eq 12
        onemingmttprimegftt = 1 - self.GMTTprime / self.f.G23
        self.G23 = self.GTTprime = eq_five(self.GMTTprime, onemingmttprimegftt)

        # eq 13
        self.nu12 = self.nu13 = self.nuTL = fvf * \
            self.f.nu12 + (1. - fvf) * self.m.nu12

        self.nu21 = self.nu31 = self.nuLT = self.nuTL * \
            self.ETprime / self.ELprime

        self.nu23 = self.nuTT = self.ETprime / (2. * self.GTTprime) - 1.

    def init_matrix_area_dens(self, mA_Fs, psi_M):
        # mA_F/(mA_F + mA_M)=(1-psi_M)
        #mA_F = (1-psi_M)*mA_F + (1-psi_M)*mA_M\

        mA_F = np.sum(mA_Fs)
        mA_M = (1 - (1 - psi_M)) * mA_F / (1 - psi_M)
        return self.mA_M

    def make_laminate(self, mA_Fs, stitch, mA_T,
                      psi_SF, rho_S, psi_M):

        self.s = stitch  # Material()
        '''
        self.pes.set_props_iso(E1=1.5E+10,
                               nu12=0.28,
                               rho=1370.0)
        '''
        #fvf = 0.549
        #self.comp = CompositeCFMh(self.f, self.m, fu)
        # self.comp.lamina_properties(fvf)

        # area densities according to Saertex UD-1200
        #mA_Fs = np.array([1.134, 0.054, 0.054, 1.134])
        # set dominant laminae for post processing
        # self.dom_lam = dom_lam  # [0, 3]
        #mA_T = 0.012
        #angles = np.array([0.0, 90.0, 90.0, 0.0])

        mA_F = np.sum(mA_Fs)
        self.mA_G = mA_F + mA_T

        #psi_SF = 0.0055
        self.mA_S = psi_SF * mA_F
        self.mA_Gl = (1 - psi_SF) * mA_F
        # (1 - (1 - psi_M)) * mA_F / (1 - psi_M)
        self.mA_M = (1 - (1 - psi_M)) * self.mA_G / (1 - psi_M)
        self.mA_tot = self.mA_G + self.mA_M

        self.psi_G = self.mA_G / self.mA_tot

        #rho_S = 1150.0
        rho_Gl = self.f.rho
        self.rho_F = mA_F / (self.mA_Gl / rho_Gl + self.mA_S / rho_S)
        self.rho_G = self.mA_G / \
            (self.mA_Gl / rho_Gl + self.mA_S / rho_S + mA_T / self.s.rho)
        self.rho_tot = self.mA_tot / \
            (self.mA_G / self.rho_G + self.mA_M / self.m.rho)

        #self.mA_M = (1 - self.fvf) * self.m.rho * self.t_tot
        #self.fvf = 1 - self.mA_M / (self.m.rho * self.t_tot)
        # self.fvf = self.psi_G * self.m.rho / \
        #    (self.psi_G * self.m.rho + (1 - self.psi_G)
        #     * self.m.rho)  # eq.3.49
        self.fvf = self.rho_tot / self.rho_G * self.psi_G

        self.t_Fs = mA_Fs / (self.rho_F * self.fvf)
        self.t_T = mA_T / (self.s.rho * self.fvf)

        self.t_F = np.sum(self.t_Fs)
        self.t_tot = self.t_F + self.t_T

        self.rho_M = self.m.rho

        self.psi_F = mA_F / self.mA_tot
        self.psi_M = self.mA_M / self.mA_tot

        # self.laminate_properties(thicknesses=t_Fs,
        #                         angles=angles,
        #                         stitch=self.pes,
        #                         t_T=t_T)

    def laminate_properties(self, thicknesses, angles, stitch, t_T):
        ''' determines laminate smeared properties
        '''
        self.pl = Plate(width=0.0)
        uniax = self.pl.add_material('uniax')

        uniax.set_props(E1=self.E1,
                        E2=self.E2,
                        E3=self.E3,
                        nu12=self.nu12,
                        nu13=self.nu13,
                        nu23=self.nu23,
                        G12=self.G12,
                        G13=self.G13,
                        G23=self.G23,
                        rho=self.rho_F)

        self.pl.s = [0]
        self.pl.init_regions(1)
        # add uniax layers to regions layup
        r = self.pl.regions['region00']
        for thickness, angle in zip(thicknesses, angles):
            l = r.add_layer('uniax')
            l.thickness = np.array([thickness])
            l.angle = np.array([angle])

        stitchingthread = self.pl.add_material('stitchingthread')
        stitchingthread.set_props_iso(E1=stitch.E1,  # 1.5E+10,
                                      nu12=stitch.nu12,  # 0.28,
                                      rho=stitch.rho
                                      )

        l = r.add_layer('stitchingthread')
        l.thickness = np.array([t_T])
        l.angle = np.array([0])

        self.pl.init_layup(ridx=0, sidx=0, lidx=[])
        self.pl.laminate.force_symmetric()
        self.pl.laminate.calc_equivalent_modulus()

    def matrix_stresses(self, sE, sR, aMT, dT, aMM, dM, aMP):
        ''' stress vector comes in notation:
        [sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]
        '''

        # eq. 18
        sRTMP = (aMT * dT + aMM * dM + aMP) * self.m.E1

        # eq. 14
        sM1 = (sE[0] + sR[0]) / self.ELprime * self.EMLprime - sRTMP

        # eq. 15
        def eq_15(u_term_t, b_term_t, c1l_term_t, c2l_term_t, c2u_term_t):
            return u_term_t / (b_term_t * ((np.sqrt(self.twosqrt3fvfpi) /
                                            c1l_term_t) + ((c2u_term_t - np.sqrt(self.twosqrt3fvfpi)) / c2l_term_t))) - sRTMP

        sM2 = eq_15(sE[1] + sR[1], self.ETprime, self.f.E2, self.EMTprime, 1)
        sM3 = eq_15(
            sE[2] + sR[2], self.ETprime, self.f.E2, self.EMTprime, np.sqrt(3))

        # eq. 16
        sM21 = eq_15(
            sE[5] + sR[5], self.GLTprime, self.f.G12, self.GMLTprime, 1)
        sM31 = eq_15(
            sE[4] + sR[4], self.GLTprime, self.f.G12, self.GMLTprime, np.sqrt(3))

        # eq. 17
        sM23 = eq_15(
            sE[3] + sR[3], self.GTTprime, self.f.G23, self.GMTTprime, 1)

        sM = np.array([sM1, sM2, sM3, sM21, sM31, sM23])

        sMe = np.sqrt(sM[0]**2 + sM[1]**2 + sM[2]**2 -
                      2 * self.m.nu12 * (sM[0] * sM[1] + sM[1] * sM[2] + sM[2] * sM[0]) +
                      2 * (1 + self.m.nu12) * (sM[3]**2 + sM[4]**2 + sM[5]**2))

        eM = sMe / self.m.s11_t
        return sM, sMe, eM

    def recover_laminate_stresses(self, elaminate_target):
        '''
        elaminate_target should contain only one non-zero entry, whihc is the strain 
        value to be obtained.
        The laminate is loaded such that this strain is obtained.

        '''
        # determine strain vector due to axial loading
        #F = np.zeros(6)
        # Fx = F[0] = 1580E+3  # unit load

        from scipy import optimize
        EPS = 1E-12

        def min_func(f):
            F = f * elaminate_target / (elaminate_target + EPS)
            eps = self.pl.laminate.apply_load(F, dT=0.)
            return abs(np.max(abs(elaminate_target)) - np.max(abs(eps)))
        f = optimize.brent(min_func)

        F = f * elaminate_target / (elaminate_target + EPS)
        elaminate = self.pl.laminate.apply_load(F, dT=0.)

        #sMs = np.zeros(len(self.pl.laminate.plies))
        sMes = np.zeros(len(self.pl.laminate.plies))
        eMs = np.zeros(len(self.pl.laminate.plies))
        for i, ply in enumerate(self.pl.laminate.plies):
            eps, sig = ply.calc_loading(elaminate)
            _, sMe, eM = self.matrix_stresses(
                sE=sig, sR=np.zeros_like(sig), aMT=0, dT=0, aMM=0, dM=0, aMP=0)
            #sMs[i] = sM
            sMes[i] = sMe
            eMs[i] = eM
        return sMes, eMs
