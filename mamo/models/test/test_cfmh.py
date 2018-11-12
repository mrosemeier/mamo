import numpy as np
import unittest
from fusedwind.turbine.layup import Material
from mamo.models.cfmh import CompositeCFMh
from pasta.plate import Plate


cfrp_ET = np.array([3.09000000e+09,   3.62312781e+09,   3.89422723e+09,
                    4.16128057e+09,   4.44301005e+09,   4.74410481e+09,
                    5.06655809e+09,   5.41163327e+09,   5.78038541e+09,
                    6.17383850e+09,   6.59305863e+09,   7.03919074e+09,
                    7.51348172e+09,   8.01729822e+09,   8.55214290e+09,
                    9.11967040e+09,   9.72170380e+09,   1.03602515e+10,
                    1.10375243e+10,   1.17559517e+10])

cfrp_GLT = np.array([1.12773723e+09,   1.27108185e+09,   1.37969507e+09,
                     1.50004601e+09,   1.63688425e+09,   1.79314841e+09,
                     1.97199416e+09,   2.17728834e+09,   2.41395734e+09,
                     2.68841713e+09,   3.00920211e+09,   3.38794504e+09,
                     3.84097683e+09,   4.39207400e+09,   5.07747422e+09,
                     5.95576863e+09,   7.12949663e+09,   8.79936992e+09,
                     1.14320038e+10,   1.65013733e+10])


cfrp_GTT = np.array([1.12773723e+09,   1.22660328e+09,   1.30460277e+09,
                     1.38710106e+09,   1.47618852e+09,   1.57245224e+09,
                     1.67624073e+09,   1.78787334e+09,   1.90769180e+09,
                     2.03607716e+09,   2.17345771e+09,   2.32031490e+09,
                     2.47718928e+09,   2.64468715e+09,   2.82348843e+09,
                     3.01435576e+09,   3.21814517e+09,   3.43581837e+09,
                     3.66845675e+09,   3.91727726e+09])
cfrp_nuTL = np.array([0.37,  0.36810526,  0.36621053,  0.36431579,  0.36242105,
                      0.36052632,  0.35863158,  0.35673684,  0.35484211,  0.35294737,
                      0.35105263,  0.34915789,  0.34726316,  0.34536842,  0.34347368,
                      0.34157895,  0.33968421,  0.33778947,  0.33589474,  0.334])
cfrp_nuLT = np.array([0.37,  0.09621643,  0.05789444,  0.04282151,  0.03487491,
                      0.03003902,  0.02683837,  0.02460394,  0.02298872,  0.02179497,
                      0.0209019,  0.02023171,  0.01973211,  0.01936679,  0.01910984,
                      0.01894238,  0.01885041,  0.01882346,  0.01885359,  0.01893479])
cfrp_nuTT = np.array([0.37,  0.47689471,  0.49249538,  0.49999185,  0.5048925,
                      0.50850522,  0.51128594,  0.51342748,  0.51502077,  0.51611113,
                      0.51672117,  0.51686108,  0.51653363,  0.5157366,  0.51446395,
                      0.51270639,  0.51045141,  0.50768323,  0.50438251,  0.50052587])

gfrp_ET = np.array([3.09000000e+09,   3.56326957e+09,   3.94255449e+09,
                    4.33884221e+09,   4.77708056e+09,   5.27008853e+09,
                    5.82871266e+09,   6.46460021e+09,   7.19156859e+09,
                    8.02679479e+09,   8.99226794e+09,   1.01168464e+10,
                    1.14393832e+10,   1.30137189e+10,   1.49170575e+10,
                    1.72648360e+10,   2.02390149e+10,   2.41468810e+10,
                    2.95585632e+10,   3.76866725e+10])
gfrp_GLT = np.array([1.12773723e+09,   1.25467904e+09,   1.37405638e+09,
                     1.50560216e+09,   1.65459816e+09,   1.82459530e+09,
                     2.01920969e+09,   2.24268601e+09,   2.50027199e+09,
                     2.79863437e+09,   3.14642823e+09,   3.55514290e+09,
                     4.04042058e+09,   4.62420478e+09,   5.33841854e+09,
                     6.23165039e+09,   7.38225360e+09,   8.92663650e+09,
                     1.11290090e+10,   1.45897069e+10])
gfrp_GTT = np.array([1.12773723e+09,   1.23122169e+09,   1.33960331e+09,
                     1.46132996e+09,   1.59978600e+09,   1.75773508e+09,
                     1.93823587e+09,   2.14500108e+09,   2.38269541e+09,
                     2.65730566e+09,   2.97667009e+09,   3.35128126e+09,
                     3.79555713e+09,   4.32994472e+09,   4.98458913e+09,
                     5.80615888e+09,   6.87163121e+09,   8.31933368e+09,
                     1.04303149e+10,   1.38970437e+10])
gfrp_nuTL = np.array([0.37,  0.36289474,  0.35578947,  0.34868421,  0.34157895,
                      0.33447368,  0.32736842,  0.32026316,  0.31315789,  0.30605263,
                      0.29894737,  0.29184211,  0.28473684,  0.27763158,  0.27052632,
                      0.26342105,  0.25631579,  0.24921053,  0.24210526,  0.235])
gfrp_nuLT = np.array([0.37,  0.18886917,  0.13226606,  0.10532727,  0.09005079,
                      0.08058328,  0.07446412,  0.07048945,  0.06801208,  0.06666591,
                      0.06624387,  0.06664049,  0.06782581,  0.0698382,  0.07279283,
                      0.07691001,  0.08257838,  0.09049361,  0.10198988,  0.11996512])
gfrp_nuTT = np.array([0.37,  0.4470463,  0.47153805,  0.48455254,  0.49303737,
                      0.49911343,  0.50361283,  0.50689905,  0.50912462,  0.51032584,
                      0.5104576,  0.50939977,  0.50694388,  0.50275809,  0.49631766,
                      0.48676917,  0.47264996,  0.4512509,  0.41695449,  0.35592408])


class MaterialTestCase(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        # micro properties as of Krimmer 2016 EASN ASSESSMENT OF QUASI-STATIC AND FATIGUE PERFORMANCE OF UNI-DIRECTIONALLY
        # FIBRE REINFORCED POLYMERS ON THE BASIS OF MATRIX EFFORT

        self.epoxy_resin = Material()
        self.epoxy_resin.set_props_iso(E1=3.09E+09,
                                       nu12=0.37,
                                       rho=1.15E+03)

        self.gfecr = Material()
        self.gfecr.set_props_iso(E1=81.5E+09,
                                 nu12=0.22,
                                 rho=2.6E+03)

        self.cfecr = Material()
        self.cfecr.set_props(E1=230E+09,
                             E2=14.7E+09,
                             E3=14.7E+09,
                             nu12=0.33,
                             nu13=0.33,
                             nu23=0.49,
                             G12=47.6E+09,
                             G13=47.6E+09,
                             G23=4.96E+09,
                             rho=999999E+03)

    def tearDown(self):
        pass

    #@unittest.skip("demonstrating skipping")
    def test_cfrp(self):

        fvf = np.linspace(0.0, 0.9, 20)

        cflamina = CompositeCFMh(self.cfecr, self.epoxy_resin, fu=1.0)
        cflamina.lamina_properties(fvf)

        self.assertEqual(
            np.testing.assert_allclose(cflamina.E2, cfrp_ET, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(cflamina.G12, cfrp_GLT, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(cflamina.G23, cfrp_GTT, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(cflamina.nu12, cfrp_nuTL, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(cflamina.nu21, cfrp_nuLT, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(cflamina.nu23, cfrp_nuTT, 1E-6), None)

    def test_gfrp(self):

        fvf = np.linspace(0.0, 0.9, 20)

        gflamina = CompositeCFMh(self.gfecr, self.epoxy_resin, fu=1.0)
        gflamina.lamina_properties(fvf)

        self.assertEqual(
            np.testing.assert_allclose(gflamina.E2, gfrp_ET, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(gflamina.G12, gfrp_GLT, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(gflamina.G23, gfrp_GTT, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(gflamina.nu12, gfrp_nuTL, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(gflamina.nu21, gfrp_nuLT, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(gflamina.nu23, gfrp_nuTT, 1E-6), None)


class MaterialResistanceTestCase(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        # source Krimmer Dissertation

        self.epoxy_resin = Material()
        self.epoxy_resin.set_props_iso(E1=3.2177E+09,
                                       nu12=0.3680,
                                       rho=1.15E+03)
        self.epoxy_resin.s11_t = 73.49E+6

        self.gfecr = Material()
        self.gfecr.set_props_iso(E1=81.383E+09,
                                 nu12=0.2210,
                                 rho=2.6E+03)

        self.pes = Material()
        self.pes.set_props_iso(E1=1.5E+10,
                               nu12=0.28,
                               rho=1370.0)

        fvf = 0.549
        self.comp = CompositeCFMh(self.gfecr, self.epoxy_resin, fu=1.0)
        self.comp.lamina_properties(fvf)

        # area densities according to Saertex UD-1200
        mA_Fs = np.array([1.134, 0.054, 0.054, 1.134])
        angles = np.array([0.0, 90.0, 90.0, 0.0])
        t_F = mA_Fs / (self.comp.f.rho * self.comp.fvf)
        mA_T = 0.012
        t_T = mA_T / (self.pes.rho * self.comp.fvf)

        mA_F = np.sum(mA_Fs)
        mA_F_meas = 2.346

        psi_F = 0.7151
        psi_M = 1 - psi_F
        mA_tot = mA_F_meas / psi_F

        psi_SF = 0.0055
        psi_S = psi_SF * mA_F
        psi_Gl = (1 - psi_SF) * mA_F
        rho_S = 1150.0
        rho_Gl = self.comp.f.rho

        rho_F = psi_F / (psi_Gl / rho_Gl + psi_S / rho_S)

        self.comp.rho_F = rho_F

        self.comp.laminate_properties(thicknesses=t_F,
                                      angles=angles,
                                      stitch=self.pes,
                                      t_T=t_T)

    @unittest.skip("check test")
    def test_tension_parallel(self):
        exx_t = 2.19E-03
        elaminate = np.array([exx_t, 0.0, 0.0, 0.0, 0.0, 0.0])

        sMes, eMs = self.comp.recover_laminate_stresses(elaminate)
        eMsmax = np.max(eMs)
        self.assertEqual(
            np.testing.assert_allclose(eMsmax, 1.0, 1E-6), None)


class MaterialDTU10MWTestCase(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        # DTU Wind Energy Report-I-0092
        # [23] M. Hinton, P. Soden, and Abdul-Salam Kaddour. Failure Criteria in Fibre-Reinforced-Polymer Composites. D Elsevier, 2004.

        self.epoxy_resin = Material()
        self.epoxy_resin.set_props_iso(E1=4.0E+09,
                                       nu12=0.35,
                                       rho=1.140E+03)
        self.epoxy_resin.s11_t = 73.49E+6

        self.eglass = Material()
        self.eglass.set_props_iso(E1=75.0E+09,
                                  nu12=0.2,
                                  rho=2.550E+03)

        self.balsa = Material()
        self.balsa.set_props(E1=0.05E+09,
                             E2=0.05E+09,
                             E3=2.73E+09,
                             nu12=0.5,
                             nu13=0.013,
                             nu23=0.013,
                             G12=0.0167E+09,
                             G13=0.150E+09,
                             G23=0.150E+09,
                             rho=110.)

        self.pes = Material()
        self.pes.set_props_iso(E1=1.5E+10,
                               nu12=0.28,
                               rho=1370.0)

        # create uniax lamina and laminate
        self.uniax = CompositeCFMh(self.eglass, self.epoxy_resin, fu=1.0)
        # lamina properties
        self.uniax.lamina_properties(fvf=0.55)

        # total mass per area of fabric (fibers incl sizing)
        mA_F = 1.2
        # mass per area of fabric's laminae
        mA_Fs = np.array([.95, .05]) * mA_F
        # angles of fabric's laminae
        angles = np.array([0.0, 90.0])
        # thicknesses of fabric's laminae
        t_F = mA_Fs / (self.uniax.f.rho * self.uniax.fvf)

        with_stitching_thread = False
        if with_stitching_thread:
            # mass per area of stitching thread
            mA_T = 0.012
            # thickness of stitching thread lamina
            t_T = mA_T / (self.pes.rho * self.uniax.fvf)
        else:
            t_T = 0.

        with_sizing = False
        if with_sizing:
            # fiber mass fraction
            psi_F = 0.7151
            # sizing mass fraction of fibers
            psi_SF = 0.0055
            # sizing mass fraction
            psi_S = psi_SF * mA_F
            # glass mass fraction
            psi_Gl = (1 - psi_SF) * mA_F
            # density of sizing
            rho_S = 1150.0
            # density of glass
            rho_Gl = self.uniax.f.rho
            # density of fabric
            rho_F = psi_F / (psi_Gl / rho_Gl + psi_S / rho_S)
        else:
            rho_F = self.uniax.f.rho
        # set fabric density
        self.uniax.rho_F = rho_F

        # laminate properties
        self.uniax.laminate_properties(thicknesses=t_F,
                                       angles=angles,
                                       stitch=self.pes,
                                       t_T=t_T)

        self.biax = CompositeCFMh(self.eglass, self.epoxy_resin, fu=1.0)
        # lamina properties
        self.biax.lamina_properties(fvf=0.5)

        # total mass per area of fabric (fibers incl sizing)
        mA_F = 0.6
        # mass per area of fabric's laminae
        mA_Fs = np.array([.5, .5]) * mA_F
        # angles of fabric's laminae
        angles = np.array([-45.0, +45.0])
        # thicknesses of fabric's laminae
        t_F = mA_Fs / (self.biax.f.rho * self.biax.fvf)

        t_T = 0.
        rho_F = self.biax.f.rho

        # set fabric density
        self.biax.rho_F = rho_F

        # laminate properties
        self.biax.laminate_properties(thicknesses=t_F,
                                      angles=angles,
                                      stitch=self.pes,
                                      t_T=t_T)

        self.triax = CompositeCFMh(self.eglass, self.epoxy_resin, fu=1.0)
        # lamina properties
        self.triax.lamina_properties(fvf=0.5)

        # total mass per area of fabric (fibers incl sizing)
        mA_F = 0.9
        # mass per area of fabric's laminae
        mA_Fs = np.array([0.3, 0.35, 0.35]) * mA_F
        # angles of fabric's laminae
        angles = np.array([0., -45., +45.])
        # thicknesses of fabric's laminae
        t_F = mA_Fs / (self.triax.f.rho * self.triax.fvf)

        t_T = 0.
        rho_F = self.triax.f.rho

        # set fabric density
        self.triax.rho_F = rho_F

        # laminate properties
        self.triax.laminate_properties(thicknesses=t_F,
                                       angles=angles,
                                       stitch=self.pes,
                                       t_T=t_T)

    def tearDown(self):
        pass

    def test_cfmh(self):
        # TODO: Add asserts

        self.uniax.E1
        self.uniax.E2
        self.uniax.E3
        self.uniax.G12
        self.uniax.G23
        self.uniax.G13
        self.uniax.nu12

        uniax_E1 = 4.163000000000000000e+10
        uniax_E1_dev = 1.0432491560828718

        self.assertEqual(
            np.testing.assert_allclose(self.uniax.E1 / uniax_E1, uniax_E1_dev, 1E-6), None)

        # materials
    # biax uniax balsa triax adhesive
    # E1 E2 E3 nu12 nu13 nu23 G12 G13 G23 rho
    # 1.392000000000000000e+10 1.392000000000000000e+10 1.209901000000000000e+10 5.330000000000000293e-01 2.750000000000000222e-01 3.328999999999999737e-01 1.150000000000000000e+10 4.538640000000000000e+09 4.538640000000000000e+09 1.845000000000000000e+03
    # 4.163000000000000000e+10 1.493000000000000000e+10 1.342583000000000000e+10 2.409999999999999920e-01 2.675000000000000155e-01 3.301000000000000045e-01 5.047000000000000000e+09 5.046980000000000000e+09 5.046980000000000000e+09 1.915500000000000000e+03
    # 5.000000000000000000e+07 5.000000000000000000e+07 2.730000000000000000e+09 5.000000000000000000e-01 1.299999999999999940e-02 1.299999999999999940e-02 1.667000000000000000e+07 1.500000000000000000e+08 1.500000000000000000e+08 1.100000000000000000e+02
    # 2.179000000000000000e+10 1.467000000000000000e+10
    # 1.209901000000000000e+10 4.779999999999999805e-01
    # 2.750000000000000222e-01 3.328999999999999737e-01
    # 9.413000000000000000e+09 4.538640000000000000e+09
    # 4.538640000000000000e+09 1.845000000000000000e+03

        '''
        >>> uniax.E1
        43430462367.72995
        >>> uniax.E2
        12836380388.803102
        >>> uniax.E3
        12836380388.803102
        >>> uniax.G12
        4654078663.4492397
        >>> uniax.G23
        4404344624.5562649
        >>> uniax.G13
        4654078663.4492397
        >>> uniax.nu12
        0.26749999999999996
        '''

        self.uniax.pl.laminate.e1
        self.uniax.pl.laminate.e2
        self.uniax.pl.laminate.g12
        self.uniax.pl.laminate.nu12
        '''
        >>> uniax.pl.laminate.e1
        41475334359.052193
        >>> uniax.pl.laminate.e2
        14411758748.890478
        >>> uniax.pl.laminate.g12
        4676103068.022542
        >>> uniax.pl.laminate.nu12
        0.23984539741893107
        '''

        self.biax.E1
        self.biax.E2
        self.biax.E3
        self.biax.G12
        self.biax.G23
        self.biax.G13
        self.biax.nu12
        self.biax.nu23
        self.biax.nu13

        '''
        >>> biax.E1
        39875728213.725014
        >>> biax.E2
        11410056461.307756
        >>> biax.E3
        11410056461.307756
        >>> biax.G12
        4116112582.1685057
        >>> biax.G23
        3906239477.590991
        >>> biax.G13
        4116112582.1685057
        >>> biax.nu12
        0.275        
        >>> biax.nu23
        0.46049116122604294
        >>> biax.nu13
        0.275
        '''

        self.biax.pl.laminate.e1
        self.biax.pl.laminate.e2
        self.biax.pl.laminate.g12
        self.biax.pl.laminate.nu12

        '''
        >>> biax.pl.laminate.e1
        12864422952.357038
        >>> biax.pl.laminate.e2
        12864422952.357037
        >>> biax.pl.laminate.g12
        11501447448.46925
        >>> biax.pl.laminate.nu12
        0.56269085156796561
        '''

        self.triax.pl.laminate.e1
        self.triax.pl.laminate.e2
        self.triax.pl.laminate.g12
        self.triax.pl.laminate.nu12

        '''
        >>> triax.pl.laminate.e1
        21196635637.131153
        >>> triax.pl.laminate.e2
        13913842366.712135
        >>> triax.pl.laminate.g12
        9285846988.5790272
        >>> triax.pl.laminate.nu12
        0.50233167427961067
        '''

    def test_stress_recovery_mechanical(self):
        pl = Plate(width=0.)

        pl.materials['uniax'] = self.uniax
        pl.materials['biax'] = self.biax
        pl.materials['triax'] = self.triax

        pl.s = [0]
        pl.init_regions(1)

        # add materials to regions layup
        r = pl.regions['region00']
        l = r.add_layer('triax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)
        l = r.add_layer('uniax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)
        l = r.add_layer('uniax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)
        l = r.add_layer('triax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)

        pl.init_layup(ridx=0, sidx=0)

        exx_t = 2190.E-06
        eps_laminate = np.array([exx_t, 0.0, 0.0, 0.0, 0.0, 0.0])

        eMsmaxs = []
        for k, v in pl.regions['region00'].layers.iteritems():
            matname = k[:-2]
            matobj = pl.materials[matname]
            _sMes, _eMs = matobj.recover_laminate_stresses(eps_laminate)
            matobj_thicks = matobj.pl.regions['region00'].thick_matrix[0]
            bool = matobj_thicks / matobj_thicks
            sMes, eMs = _sMes * bool, _eMs * bool

            eMsmax = np.nanmax(eMs)
            eMsmaxs.append(eMsmax)

        np.asarray(eMsmaxs)

    def test_stress_recovery_thermal(self):
        pl = Plate(width=0.)

        pl.materials['uniax'] = self.uniax
        pl.materials['biax'] = self.biax
        pl.materials['triax'] = self.triax

        pl.s = [0]
        pl.init_regions(1)

        # add materials to regions layup
        r = pl.regions['region00']
        l = r.add_layer('triax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)
        l = r.add_layer('uniax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)
        l = r.add_layer('uniax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)
        l = r.add_layer('triax')
        l.thickness = np.array([8.000000000000000167e-03])
        l.angle = np.zeros(1)

        pl.init_layup(ridx=0, sidx=0)

        dT = 60.
        eps_laminate = pl.laminate.apply_load(F=0., dT=dT)

        eMsmaxs = []
        for k, v in pl.regions['region00'].layers.iteritems():
            matname = k[:-2]
            matobj = pl.materials[matname]
            _sMes, _eMs = matobj.recover_laminate_stresses(eps_laminate)
            matobj_thicks = matobj.pl.regions['region00'].thick_matrix[0]
            bool = matobj_thicks / matobj_thicks
            sMes, eMs = _sMes * bool, _eMs * bool

            eMsmax = np.nanmax(eMs)
            eMsmaxs.append(eMsmax)

        np.asarray(eMsmaxs)


if __name__ == "__main__":
    unittest.main()
