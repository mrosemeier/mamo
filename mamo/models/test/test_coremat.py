import numpy as np
import unittest
from fusedwind.turbine.layup import Material
from mamo.models.coremat import CoreMaterial

core_inf_E1 = 193333333.3333333
core_inf_E2 = 50885065.9272254
core_inf_G12 = 44185957.26693017
core_inf_G23 = 18173543.21452971
core_inf_G13 = 70198371.31933063
core_inf_nu12 = 0.3985164835164835


class CoreMatTestCase(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        # micro properties as of Krimmer 2016 EASN ASSESSMENT OF QUASI-STATIC AND FATIGUE PERFORMANCE OF UNI-DIRECTIONALLY
        # FIBRE REINFORCED POLYMERS ON THE BASIS OF MATRIX EFFORT

        self.epoxy_resin = Material()
        self.epoxy_resin.set_props_iso(E1=3.09E+09,
                                       nu12=0.37,
                                       rho=1.15E+03)

        self.core_dry = Material()
        self.core_dry.set_props_iso(E1=4.85E+07,  # PVC core
                                    nu12=0.4000,
                                    rho=8.00E+01,)
        # G12 = 1.73E+07Pa; however Haselbach et al 2016 uses G12 = 3.91E+07 Pa

        self.alpha_x = 0.
        self.alpha_y = 0.2 / 4.2

    def tearDown(self):
        pass

    # @unittest.skip("demonstrating skipping")
    def test_coremat(self):

        coremat = CoreMaterial(self.core_dry, self.epoxy_resin, self.alpha_x,
                               self.alpha_y)
        core_inf = coremat.calc_infused_props()

        self.assertEqual(
            np.testing.assert_allclose(core_inf.E1, core_inf_E1, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(core_inf.E2, core_inf_E2, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(core_inf.G12, core_inf_G12, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(core_inf.G23, core_inf_G23, 1E-6), None)
        self.assertEqual(
            np.testing.assert_allclose(core_inf.G13, core_inf_G13, 1E-6), None)    
        self.assertEqual(
            np.testing.assert_allclose(core_inf.nu12, core_inf_nu12, 1E-6), None)


if __name__ == "__main__":
    unittest.main()
