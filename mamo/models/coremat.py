from fusedwind.turbine.layup import Material


class CoreMaterial(object):
    ''' Implementation of the slit infused core material model by Nickel 2014
    '''
    
    def __init__(self, core_dry, matrix, alpha_x, alpha_y):
        '''
        :param: core_dry: Fused-wind Material object of dry core
        :param: matrix: Fused-wind Material object of matrix
        :param: alpha_x: slit width / distance in x-direction
        :param: alpha_y: slit width / distance in y-direction
        '''

        self.core_dry = core_dry
        self.matrix = matrix
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
    
    def calc_infused_props(self):
        '''
        :return: core_inf: Fused-wind Material object of infused core
        '''
    
        Ec = self.core_dry.E1
        Er = self.matrix.E1
        Ec_Er = Ec / Er
        Gc = self.core_dry.G12
        Gr = self.matrix.G12
        Gc_Gr = Gc / Gr
        # Nickels equations (2.29)
        Ex_ni = self.alpha_y * Er + (1 - self.alpha_y) * Ec / \
            ((self.alpha_y + (1 - self.alpha_y) * Ec_Er)
             * self.alpha_x + (1 - self.alpha_x))
        # (2.31)
        Ey_ni = self.alpha_x * Er + (1 - self.alpha_x) * Ec / \
            ((self.alpha_x + (1 - self.alpha_x) * Ec_Er)
             * self.alpha_y + (1 - self.alpha_y))
        # (2.34)
        Ez_ni = (1 - self.alpha_x) * (1 - self.alpha_y) * Ec + \
            (1 - (1 - self.alpha_x) * (1 - self.alpha_y)) * Er
        # (2.43)
        Gxz_ni = self.alpha_y * Gr + (1 - self.alpha_y) * Gc / \
            ((self.alpha_y + (1 - self.alpha_y) * Gc_Gr)
             * self.alpha_x + (1 - self.alpha_x))
        # (2.45)
        Gyz_ni = self.alpha_x * Gr + (1 - self.alpha_x) * Gc / \
            ((self.alpha_x + (1 - self.alpha_x) * Gc_Gr)
             * self.alpha_y + (1 - self.alpha_y))
    
        # (2.54)
        Gyx_ni = self.alpha_x * Gr + (1 - self.alpha_x) * Gc / \
            ((self.alpha_x + (1 - self.alpha_x) * Gc_Gr)
             * self.alpha_y + (1 - self.alpha_y))
        # (2.59)
        Gxy_ni = self.alpha_y * Gr + (1 - self.alpha_y) * Gc / \
            ((self.alpha_y + (1 - self.alpha_y) * Gc_Gr)
             * self.alpha_x + (1 - self.alpha_x))
    
        # (2.60)
        Gxym_ni = 0.5 * (Gxy_ni + Gyx_ni)
    
        nur = self.matrix.nu12
        nuc = self.core_dry.nu12
        nuc_nur = nuc / nur
    
        nuxz_ni = self.alpha_y * nur + (1 - self.alpha_y) * nuc / \
            ((self.alpha_y + (1 - self.alpha_y) * nuc_nur)
             * self.alpha_x + (1 - self.alpha_x))
        nuyz_ni = self.alpha_x * nur + (1 - self.alpha_x) * nuc / \
            ((self.alpha_x + (1 - self.alpha_x) * nuc_nur)
             * self.alpha_y + (1 - self.alpha_y))
        nuyx_ni = self.alpha_x * nur + (1 - self.alpha_x) * nuc / \
            ((self.alpha_x + (1 - self.alpha_x) * nuc_nur)
             * self.alpha_y + (1 - self.alpha_y))
        nuxy_ni = self.alpha_y * nur + (1 - self.alpha_y) * nuc / \
            ((self.alpha_y + (1 - self.alpha_y) * nuc_nur)
             * self.alpha_x + (1 - self.alpha_x))
        nuxym_ni = 0.5 * (nuxy_ni + nuyx_ni)
        
        # TODO: obtain density of infused core
        rho_ni = 0.
        core_inf = Material()
        core_inf.set_props(E1=Ex_ni,
                           E2=Ey_ni,
                           E3=Ez_ni,
                           nu12=nuxym_ni,
                           nu13=nuxz_ni,
                           nu23=nuyz_ni,
                           G12=Gxym_ni,
                           G13=Gxz_ni,
                           G23=Gyz_ni,
                           rho=rho_ni)
        return core_inf
