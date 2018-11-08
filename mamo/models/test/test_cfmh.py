import numpy as np
import unittest
from fusedwind.turbine.layup import Material
from mamo.models.cfmh import CompositeCFMh


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


if __name__ == "__main__":
    unittest.main()
