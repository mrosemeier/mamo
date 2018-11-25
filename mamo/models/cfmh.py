import numpy as np
from pasta.core.plate import Plate


class CompositeCFMh(object):
    ''' Implementation of the CFM_h (german ZFM_h) model as of Krimmer 2016
    '''

    def __init__(self, fiber, matrix, fu=1.0):
        '''
        Initialize fiber and matrix objects.
        :param: fiber: Fused-wind material object
        :param: matrix: Fused-wind material object
        :param: fu: undulation reduction factor (float)
        '''
        self.f = fiber
        self.m = matrix

        # ondulation correction for stiffness and poisson's ratio
        # according to Krimmer 2014, eq. 3.1
        self.f.E1 = fu * self.f.E1
        self.f.nu12 = self.f.nu13 = fu * self.f.nu12

        # init input material properties for post-processing
        self.f_E1 = self.f.E1
        self.f_E2 = self.f.E2
        self.f_E3 = self.f.E3
        self.f_G12 = self.f.G12
        self.f_G23 = self.f.G23
        self.f_G13 = self.f.G13
        self.f_nu12 = self.f.nu12
        self.f_nu23 = self.f.nu23
        self.f_nu13 = self.f.nu13
        self.f_cte1 = self.f.cte1
        self.f_cte2 = self.f.cte2

        self.m_E1 = self.m.E1
        self.m_E2 = self.m.E2
        self.m_E3 = self.m.E3
        self.m_G12 = self.m.G12
        self.m_G23 = self.m.G23
        self.m_G13 = self.m.G13
        self.m_nu12 = self.m.nu12
        self.m_nu23 = self.m.nu23
        self.m_nu13 = self.m.nu13
        self.m_cte1 = self.m.cte1
        self.m_cte2 = self.m.cte2

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
        # self.psi_M = psi_M
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
        # self.psi_T = 1 - self.psi_Gl - psi_M

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
        # self.m.rho = self.rho_M

        # layer thicknesses
        # self.t_Gl = self.mA_Gl / (rho_Gl * self.fvf)
        self.t_f = self.mA_G / (self.rho_G * self.fvf)
        self.t_T = mA_T / (self.rho_T * self.fvf)

    def lamina_properties(self, fvf):
        ''' use either a random vector for fvf or self.fvf
        :param: fvf: fiber volume fraction (float)
        '''
        self.fvf = fvf

        sqrt3 = np.sqrt(3.)
        twosqrt3fvfpi = 2. * sqrt3 * fvf / np.pi
        self.twosqrt3fvfpi = twosqrt3fvfpi
        oneminemeft = 1. - self.m.E2 / self.f.E2

        # eq 5
        def eq_five(first_term_t, oneminfrac):
            result = 2. * first_term_t / sqrt3 * (sqrt3 / 2. - np.sqrt(twosqrt3fvfpi) - np.pi / (2. * oneminfrac) + ((np.pi / 2. + np.arctan(
                (np.sqrt(twosqrt3fvfpi) * oneminfrac) / np.sqrt(1. - twosqrt3fvfpi * oneminfrac ** 2))) / (oneminfrac * np.sqrt(1. - twosqrt3fvfpi * oneminfrac ** 2))))
            return result

        self.ET = eq_five(self.m.E1, oneminemeft)

        # eq. 6
        self.nuMTprime = (1. - (self.f.nu12 + self.f.nu23) * self.m.E1 /
                          (self.m.nu23 * (self.f.E1 + self.f.E2))) * self.m.nu23

        A = self.ET / self.m.E2 * \
            (1. - self.nuMTprime ** 2) + self.nuMTprime ** 2

        self.EMTprime = (self.f.E1 * A * fvf + self.ET * (1. - fvf)) / \
            (self.f.E1 / self.m.E1 * (self.ET / self.m.E2 * (1. - 3. * self.nuMTprime ** 2 - 2. * self.nuMTprime ** 3) +
                                      2. * (self.nuMTprime ** 2 + self.nuMTprime ** 3)) * fvf +
             A * (1. - fvf))

        # eq. 7
        self.nuMLprime = (1. - (self.f.nu12 * self.m.E1) /
                          (self.m.nu12 * self.f.E2)) * self.m.nu12

        self.EMLprime = (self.ET * (1. - self.nuMLprime) + self.m.E1 * self.nuMLprime) / \
            (self.ET / self.m.E1 * (1. - self.nuMLprime) + self.nuMLprime +
             2. * self.nuMLprime ** 2 * (1. - self.ET / self.m.E1))

        # eq 8
        self.E1 = self.ELprime = self.f.E1 * \
            fvf + self.EMLprime * (1. - fvf)

        oneminemtprimeeft = 1. - self.EMTprime / self.f.E2
        # eq 9
        self.E2 = self.E3 = self.ETprime = eq_five(
            self.EMTprime, oneminemtprimeeft)

        # eq 10
        self.GMLTprime = (self.EMTprime + self.EMLprime) / \
            (4. * (1. + self.m.nu12))

        self.nuMTTprime = self.m.nu23 * (1 + fvf * (self.f.E1 / self.m.E1 * (1 + self.nuMTprime) - 1.)) / \
            (1. + fvf * (self.f.E1 / self.m.E1 *
                         (1. + self.nuMTprime ** 2 * (self.m.E1 / self.ET - 1.)) - 1.))

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

        self._determine_cte()

    def _determine_cte(self):
        ''' smear ctes of the lamina
        '''

        if not self.f.cte1:
            self.f.cte1 = 0.
        if not self.f.cte2:
            self.f.cte2 = 0.
        if not self.m.cte1:
            self.m.cte1 = 0.

        fvf = self.fvf
        cte_M = self.m.cte1
        cte_FL = self.f.cte1
        cte_FT = self.f.cte2
        EFL = self.f.E1
        EFT = self.f.E2

        use_krimmer = True
        if use_krimmer:
            # longitudinal according to Krimmer 201X Hygro-thermal behaviour of
            # unidirectionally fibre reinforced polymer matrix composites
            self.cte1 = (fvf * EFL * cte_FL + (1 - fvf) *
                         self.EMLprime * cte_M) / self.ELprime
            self.cte2 = (fvf * EFT * cte_FT + (1 - fvf) *
                         self.EMTprime * cte_M) / self.ETprime
        else:
            EM = self.m.E1
            nuM = self.m.nu12
            # longitudinal according to Schuermann eq. 12.9
            self.cte1 = (cte_M * EM * (1 - fvf) + cte_FL * EFL * fvf) / \
                (EM * (1 - fvf) + EFL * fvf)

            simple_cte2 = True
            if simple_cte2:
                # transverse according to Schuermann eq. 12.10
                self.cte2 = fvf * cte_FT + (1 - fvf) * cte_M
            else:
                # transverse according to Schneider (Schuermann eq. 12.12)
                self.cte2 = cte_M - (cte_M - cte_FT) * \
                    ((2 * (nuM ** 3 + nuM ** 2 - nuM - 1) * 1.1 * fvf) /
                     (1.1 * fvf * (2 * nuM ** 2 + nuM - 1) - (1 + nuM)) -
                     (nuM * EFT / EM) / (EFT / EM + (1 - 1.1 * fvf) /
                                         (1.1 * fvf)))

        self.cte3 = self.cte2

    def laminate_properties(self, mA_Fs, angles, mA_T, psi_SF, rho_S, stitch):
        ''' Calc of laminate properties
        :param: mA_Fs: vector of fiber area density per lamina in kg/m**2 (np.array)
        :param: angles: vector of fiber angles per lamina in deg (np.array)
        :param: psi_SF: mass fraction of sizing per fiber mass (float)
        :param: rho_S: Density of sizing (float)
        :param: mA_T: area density of stitching thread in kg/m**2 (float)
        :param: stitch: Fused-wind material object for stitching thread
        '''
        self.s = stitch
        # lamina fiber densities
        self.mA_Fs = mA_Fs
        # fiber area density
        self.mA_F = mA_F = np.sum(mA_Fs)
        # stitching thread density
        self.mA_T = mA_T
        # thickness of laminas (Eq. 3.52)
        self.t_Fs = self.mA_Fs / (self.f.rho * self.fvf)
        # thickness of laminas (Eq. 3.52)
        self.t_F = self.mA_F / (self.f.rho * self.fvf)
        # thickness of stitching thread lamina (Eq. 3.52)
        self.t_T = self.mA_T / (self.s.rho * self.fvf)
        # thickness of laminate
        self.t_tot = self.t_F + self.t_T
        # area density of matrix (derived from Eq. 3.52)
        self.mA_M = (1 - self.fvf) * self.m.rho * self.t_tot
        # area density of fabric
        mA_G = mA_F + mA_T
        # area density of fiber (w/o sizing)
        self.mA_Gl = mA_G * (1 - psi_SF) - mA_T
        # area density of sizing
        self.mA_S = mA_F * psi_SF
        # laminate area density
        self.mA_tot = mA_G + self.mA_M
        # matrix mass fraction
        self.psi_M = self.mA_M / self.mA_tot
        # fiber mass fraction(w/o sizing)
        self.psi_Gl = self.mA_Gl / self.mA_tot
        # sizing mass fraction
        self.psi_S = self.psi_Gl * psi_SF / (1 - psi_SF)
        # density of fiber (w/o sizing)
        self.rho_Gl = self.f.rho
        # fiber mass fraction (with sizing)
        psi_F = self.psi_Gl + self.psi_S
        # density of sizing
        self.rho_S = rho_S
        # density of fiber (with sizing)
        rho_F = psi_F / (self.psi_Gl / self.rho_Gl + self.psi_S / self.rho_S)
        # stitching thread mass fraction
        self.psi_T = mA_T / mA_F * psi_F
        # fabric mass fraction
        psi_G = psi_F + self.psi_T
        # matrix mass fraction
        psi_M = 1 - psi_G
        # density of matrix
        self.rho_M = self.m.rho
        # stitching thread density
        self.rho_T = self.s.rho
        # density of fabric
        self.rho_G = psi_G / (self.psi_Gl / self.rho_Gl +
                              self.psi_S / rho_S + self.psi_T / self.rho_T)
        # fabric mass fraction (fiber + sizing + stitching)
        # derived from corrected! Eq.3.49
        psi_G = self.fvf * self.rho_G / \
            (self.fvf * (self.rho_G - self.rho_M) + self.rho_M)

        # density of laminate
        self.rho_tot = 1 / (self.psi_Gl / self.rho_Gl + self.psi_S / self.rho_S +
                            self.psi_T / self.rho_T + psi_M / self.rho_M)

        # set fabric density
        self.rho_F = rho_F

        self.angles = angles

        self._smeared_properties()

        # init stitching thread properties
        self.s_E1 = self.s.E1
        self.s_G12 = self.s.G12
        self.s_nu12 = self.s.nu12
        self.s_cte1 = self.s.cte1

    def write_laminate_properties(self, filename):
        ''' Writes a list of laminate properties to file
        '''
        fmsprop_list = ['f_E1',
                        'f_E2',
                        'f_G12',
                        'f_nu12',
                        'f_cte1',
                        'm_E1',
                        'm_G12',
                        'm_nu12',
                        'm_cte1',
                        's_E1',
                        's_G12',
                        's_nu12',
                        's_cte1',
                        ]

        self._write_variables(filename, fmsprop_list,
                              'w', 'fiber, matrix, and stitching thread properties')

        laminaprop_list = ['E1',
                           'E2',
                           'E3',
                           'G12',
                           'G23',
                           'G13',
                           'nu12',
                           'nu23',
                           'nu13',
                           'cte1',
                           'cte2',
                           ]

        self._write_variables(filename, laminaprop_list,
                              'a', 'lamina stiffness properties')

        massprop_list = ['fvf',
                         'mA_Gl',
                         'mA_S',
                         'mA_T',
                         'mA_M',
                         'mA_tot',
                         'psi_Gl',
                         'psi_S',
                         'psi_T',
                         'psi_M',
                         'rho_Gl',
                         'rho_S',
                         'rho_T',
                         'rho_M',
                         'rho_tot',
                         't_Fs',
                         't_T',
                         't_tot'
                         ]

        self._write_variables(filename, massprop_list,
                              'a', 'laminate mass properties')

        laminateprop_list = ['Ex',
                             'Ey',
                             'Ez',
                             'Gxy',
                             'Gyz',
                             'Gxz',
                             'nuxy',
                             'nuyz',
                             'nuxz',
                             'ctex',
                             'ctey',
                             ]

        self._write_variables(filename, laminateprop_list,
                              'a', 'laminate stiffness properties')

    def _write_variables(self, filename, variable_list, mode='w', header=''):
        ''' Writes a list of property names to a text file in the manner:
        variable_name = variable_value
        The function handles str, bool, int, floats. If a variable is not
        existing it is skipped.
        :param filename: the file to write the variables
        :param variable_list: the list of variable names
        :param mode: 'w' = write or 'a'= append
        :param header: header line
        '''
        with open(filename, mode) as write_file:
            if header:
                write_file.write('# ' + header + '\n')
            for k in variable_list:
                try:
                    v = getattr(self, k)
                except:
                    print 'Attribute %s not found' % k
                    v = None
                k_str = k  # .upper()  # upper case for APDL conformity
                if isinstance(v, bool):
                    v_str = str(int(v))
                elif isinstance(v, str):
                    v_str = "'" + v + "'"  # .upper()
                else:
                    v_str = str(v)
                write_file.write(k_str + ' = ' + v_str + '\n')

    def _smeared_properties(self):
        ''' Creates a PASTA plate object and determines laminate smeared properties
        '''
        self.pl = Plate(width=0.0)
        self.udlayer_name = 'udlayer'
        udlayer = self.pl.add_material(self.udlayer_name)

        udlayer.set_props(E1=self.E1,
                          E2=self.E2,
                          E3=self.E3,
                          nu12=self.nu12,
                          nu13=self.nu13,
                          nu23=self.nu23,
                          G12=self.G12,
                          G13=self.G13,
                          G23=self.G23,
                          rho=self.rho_F,
                          cte1=self.cte1,
                          cte2=self.cte2,
                          cte3=self.cte3
                          )

        self.pl.s = [0]
        self.pl.init_regions(1)
        # add udlayer layers to regions layup
        r = self.pl.regions['region00']
        for thickness, angle in zip(self.t_Fs, self.angles):
            l = r.add_layer(self.udlayer_name)
            l.thickness = np.array([thickness])
            l.angle = np.array([angle])

        self.stitchlayer_name = 'stitchingthread'
        stitchingthread = self.pl.add_material(self.stitchlayer_name)
        stitchingthread.set_props_iso(E1=self.s.E1,
                                      nu12=self.s.nu12,
                                      rho=self.s.rho,
                                      cte1=self.cte1
                                      )

        l = r.add_layer(self.stitchlayer_name)
        l.thickness = np.array([self.t_T])
        l.angle = np.array([0])

        self.pl.init_layup(ridx=0, sidx=0, lidx=[])
        self.pl.laminate.force_symmetric()
        self.pl.laminate.calc_equivalent_modulus()

        # assign smeared properties to cpmh object
        self.Ex = self.pl.laminate.e1
        self.Ey = self.pl.laminate.e2
        self.Gxy = self.pl.laminate.g12
        self.nuxy = self.pl.laminate.nu12
        self.ctex = self.pl.laminate.a1
        self.ctey = self.pl.laminate.a2
        self.ctexy = self.pl.laminate.a12

        # derived properties from lamina (neglecting stitching thread layer
        # stiffness)
        self.Ez = self.E3
        use_schuerman = False
        if use_schuerman:
            # Schuermann, p.202, eq. 8.35
            self.Gyz = self.E3 / (2 * (1 + self.nu13))
        else:
            # take from lamina
            self.Gyz = self.G23
        self.nuyz = self.nu23

        # set 13 and 12 planes equal
        self.Gxz = self.Gxy
        self.nuxz = self.nuxy

    def recover_laminate_stresses(self, elaminate, dT=0.):
        ''' Recover stress of each lamina by a given strain vector.
        :param: elaminate: [eps_x, eps_y, eps_z, gamma_yz, gamma_xz, gamma_xy]
        :param: dT: temperatur difference
        '''
        self.nlay = len(self.pl.laminate.plies)
        sMs = np.zeros((self.nlay, 6))
        sMes = np.zeros(self.nlay)
        eMs = np.zeros(self.nlay)
        # determine lamina strain
        eps0, eps1 = self.pl.laminate.apply_load(F=0., dT=dT)
        # [eps_x, eps_y, eps_z, gamma_yz, gamma_xz, gamma_xy]
        eps_lamina = np.zeros(6)
        eps_lamina[0:2] += eps0[0:2]
        eps_lamina[5] += eps0[2]
        include_thermal_bending = False
        if include_thermal_bending:
            eps_lamina[0:2] += eps1[0:2]
            eps_lamina[5] += eps1[2]
        # delta strain
        deps = eps_lamina - elaminate

        for i, (ply, lamina_name)in enumerate(zip(self.pl.laminate.plies,
                                                  self.pl.regions['region00'].layers.iterkeys())):
            layer_thick = self.pl.regions['region00'].thick_matrix[0, i]
            if layer_thick > 0.:
                if not lamina_name[:-2] == self.stitchlayer_name:
                    _, sig = ply.calc_loading(deps, dT)
                    sM, sMe, eM, _ = self.matrix_stresses(
                        sE=-sig, sR=np.zeros_like(sig), dT=-dT, neglect_sRTMP3=True)
                else:
                    sM, sMe, eM = np.zeros(6), 0., 0.
            else:
                sM, sMe, eM = np.zeros(6), 0., 0.
            sMs[i, :] = sM
            sMes[i] = sMe
            eMs[i] = eM
        return sMs, sMes, eMs

    def recover_laminate_stresses_coupon_test(self, elaminate_target):
        '''
        elaminate_target should contain only one non-zero entry, whihc is the strain 
        value to be obtained.
        The laminate is loaded such that this strain is obtained.
        '''

        # determine strain vector due to axial loading
        from scipy import optimize
        EPS = 1E-12

        def min_func(f):
            F = f * elaminate_target / (elaminate_target + EPS)
            eps = self.pl.laminate.apply_load(F, dT=0.)
            return abs(np.max(abs(elaminate_target)) - np.max(abs(eps)))

        f = optimize.brent(min_func)

        F = f * elaminate_target / (elaminate_target + EPS)
        elaminate = self.pl.laminate.apply_load(F, dT=0.)

        return self.recover_laminate_stresses(elaminate)

    def matrix_stresses(self, sE, sR, dT=0., dM=0., aMP=0., g_mat_static=1.0, neglect_sRTMP3=False):
        ''' stress vector comes in notation:
        [sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]
        :param: sE: external stress
        :param: sR: residual stress
        :param: dT: temperature difference
        :param: dM: mass difference due to moisture effects
        :param: aMP: polymer-physical expansion
        :param: g_mat_static: fatigue safety factor for matrix
        :return: sM: matrix stress vector
        :return: sMe: matrix equivalent stress
        :return: eM: matrix stress exposure (effort)
        :return: mode: matrix failure mode
        '''

        # eq. 18
        aMT = self.m.cte1
        aMM = 0.  # moisture effects not yet implemented
        sRTMP = (aMT * dT + aMM * dM + aMP) * self.m.E1

        # eq. 14
        sM1 = (sE[0] + sR[0]) / self.ELprime * self.EMLprime - sRTMP

        # eq. 15
        def eq_15(u_term_t, b_term_t, c1l_term_t, c2l_term_t, c2u_term_t):
            return u_term_t / (b_term_t * ((np.sqrt(self.twosqrt3fvfpi) /
                                            c1l_term_t) + ((c2u_term_t - np.sqrt(self.twosqrt3fvfpi)) / c2l_term_t)))

        sM2 = eq_15(sE[1] + sR[1], self.ETprime,
                    self.f.E2, self.EMTprime, 1) - sRTMP

        if neglect_sRTMP3:
            sM3 = eq_15(sE[2] + sR[2], self.ETprime,
                        self.f.E2, self.EMTprime, np.sqrt(3))
        else:
            sM3 = eq_15(sE[2] + sR[2], self.ETprime,
                        self.f.E2, self.EMTprime, np.sqrt(3)) - sRTMP

        # eq. 16
        sM21 = eq_15(
            sE[5] + sR[5], self.GLTprime, self.f.G12, self.GMLTprime, 1)
        sM31 = eq_15(
            sE[4] + sR[4], self.GLTprime, self.f.G12, self.GMLTprime, np.sqrt(3))

        # eq. 17
        sM23 = eq_15(
            sE[3] + sR[3], self.GTTprime, self.f.G23, self.GMTTprime, 1)

        # matrix stress vector
        sM = np.array([sM1, sM2, sM3, sM21,  sM23, sM31])
        # equivalent stress
        sMe, mode = self.sigma_beltrami(self.m.nu12, sM)
        # stress expusure
        eM = g_mat_static * sMe / self.m.s11_t
        return sM, sMe, eM, mode

    @staticmethod
    def sigma_beltrami(nu, sM):
        ''' Calculates equivalent stress of the matrix according to Beltrami, 1885, 
        Sulle condizioni di resistenza dei corpi elastici
        :param: nu: Poisson's ratio of matrix
        :param: sM: matrix stress vector np.array([sM1, sM2, sM3, sM21,  sM23, sM31])
        :return: sMe: equivalent stress
        :return: mode: failure mode
        '''
        sigma11 = sM[0]
        sigma22 = sM[1]
        sigma33 = sM[2]
        tau12 = sM[3]
        tau23 = sM[4]
        tau13 = sM[5]

        # check if normal tension or compression
        sign = np.sign((sigma11 + sigma22 + sigma33) / 3.)
        # correct sign in case all sigmas are zero we set sign negative to check
        # failure mode
        if sign.any() == 0:
            sign = 1.0
            check_mode = True
        elif sign.any() == 1.0:
            mode = 'A'  # tension
            check_mode = False
        else:
            check_mode = True
        if check_mode:
            # check which part is contributing most
            sMe_tc = np.sqrt(sigma11**2 + sigma22**2 + sigma33**2 -
                             2 * nu * (sigma11 * sigma22 + sigma22 * sigma33 + sigma33 * sigma11))
            sMe_s = np.sqrt(
                2 * (1 + nu) * (tau12**2 + tau23**2 + tau13**2))
            if np.max(sMe_tc) < np.max(sMe_s):
                mode = 'B'  # shear
            else:
                mode = 'C'  # compression

        sMe = sign * np.sqrt(sigma11**2 + sigma22**2 + sigma33**2 -
                             2 * nu * (sigma11 * sigma22 + sigma22 * sigma33 + sigma33 * sigma11) +
                             2 * (1 + nu) * (tau12**2 + tau23**2 + tau13**2))
        return sMe, mode

    def fatigue_stress_exposure_incr(self,
                                     m,
                                     sMe_mi,
                                     sMe_ai,
                                     ni,
                                     g_load=1.0,
                                     g_mat_static=1.0,
                                     g_mat_fat=1.0):
        ''' Calculate fatigue stress exposure of matrix for load increment
        :param: m: S/N-curve slope of matrix
        :param: sMe_mi: mean stress
        :param: sMe_ai: stress amplitude
        :param: ni: load cycles
        :param: g_load: safety factor for loads
        :param: g_mat_static: fatigue safety factor for matrix
        :param: g_mat_fat: static safety factor for matrix
        :return: eM_mi: mean stress exposure
        :return: eM_ai: amplitude stress exposure
        :return: eM_fi: fatigue stress exposure
        :return: Di: damage of increment
        '''

        # upper and lower eq stress
        sMe_ui = sMe_mi + sMe_ai
        sMe_li = sMe_mi - sMe_ai
        # upper and lower effort
        eM_ui = g_load * sMe_ui / self.m.s11_t
        eM_li = g_load * sMe_li / self.m.s11_t

        # mean effort Krimmer 2018 IQPC
        eM_mi = g_mat_static * 0.5 * abs(eM_ui + eM_li)
        eM_ai = g_mat_fat * 0.5 * abs(eM_ui - eM_li)

        # allowable cycles
        if eM_ui <= 1:
            Ni = ((1 - eM_mi) / (eM_ai))**m

        elif eM_ui > 1:
            # Krimmer degradation IQPC 2018 (no real derivation, just to
            # overcome poles)
            Ni = 1. / eM_ui
        # damage increment of a bin
        Di = ni / Ni
        # damage increment of a bin
        eM_fi = Di**(1.0 / m)

        return eM_mi, eM_ai, eM_fi, Di

    def fatigue_stress_exposure_total(self, m, Di):
        ''' Calculate fatigue stress exposure of matrix for load increment
        :param: m: S/N-curve slope of matrix
        :param: Di: damage of increment
        :return: eM_f: total fatigue stress exposure
        :return: D: total damage sum
        '''

        # total damage
        D = np.sum(Di)
        # stress exposure
        eM_f = D**(1.0 / m)

        return eM_f, D
