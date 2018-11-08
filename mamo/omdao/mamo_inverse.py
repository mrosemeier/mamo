import os
import numpy as np
from fusedwind.turbine.layup import Material
from openmdao.core.component import Component
from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.recorders.sqlite_recorder import SqliteRecorder
import sqlitedict
import tabulate
from collections import OrderedDict
import copy
from pylatex import Document, Section, Subsection, Command, Itemize, NoEscape
from mamo.models.cfmh import CompositeCFMh


class CompositeCFMhComponent(Component):

    def __init__(self, cfg):
        super(CompositeCFMhComponent, self).__init__()

        for k, w in cfg.iteritems():
            try:
                setattr(self, k, w)
            except:
                pass

        #self.add_param('fu', val=0.0)
        #self.add_param('fkor', val=0.0)
        #self.add_param('rho_Gl', val=0.0)
        #self.add_param('rho_R', val=0.0)
        #self.add_param('EF', val=0.0)
        #self.add_param('EM', val=0.0)
        #self.add_param('nuF', val=0.0)
        #self.add_param('nuM', val=0.0)
        self.add_param('frac_0', val=0.0)
        self.add_param('frac_90', val=0.0)
        self.add_param('mA_T', val=0.0)
        #self.add_param('fvf', val=0.0)
        #self.add_param('rho_T', val=0.0)

        self.add_output('xi_M', val=0.0)
        self.add_output('rho_M', val=0.0)
        self.add_output('rho_G', val=0.0)
        self.add_output('fvf', val=0.0)
        self.add_output('E1_r', val=0.0)
        self.add_output('E2_r', val=0.0)
        self.add_output('G12_r', val=0.0)
        self.add_output('master_r', val=0.0)

        self.add_output('t_f', val=0.0)
        self.add_output('t_T', val=0.0)
        self.add_output('frac_thicknesses', np.zeros(4))
        self.add_output('mA_Fs', np.zeros(4))
        self.add_output('mA_M', val=0.0)
        self.add_output('mA_S', val=0.0)
        self.add_output('mA_G', val=0.0)
        self.add_output('mA_tot', val=0.0)

        self.add_output('sMemax', np.zeros(4))

    def solve_nonlinear(self, params, unknowns, resids):
        # fu = params['fu']  # ondulation correction factor
        # fkor = params['fkor']  # porosity correction factor
        # rho_Gl = params['rho_Gl']  # fiber density
        # rho_R = params['rho_R']  # resin density
        # EF = params['EF']  # Young's modulus fiber
        # EM = params['EM']  # Young's modulus matrix
        # nuF = params['nuF']  # poisson's ratio fiber
        # nuM = params['nuM']  # poisson's ratio matrix
        frac_0 = params['frac_0']
        frac_90 = params['frac_90']
        mA_T = params['mA_T']  # stitching thread grammage
        #fvf = params['fvf']
        # rho_T = params['rho_T']  # sewing thread density

        fu = self.fu
        fkor = self.fkor
        rho_Gl = self.rho_Gl
        rho_R = self.rho_R
        EF = self.EF
        EM = self.EM
        nuF = self.nuF
        nuM = self.nuM
        rho_T = self.rho_T

        resin = Material()
        resin.set_props_iso(E1=EM,
                            nu12=nuM,
                            rho=rho_R)

        gfe = Material()
        gfe.set_props_iso(E1=EF,
                          nu12=nuF,
                          rho=rho_Gl)

        comp = CompositeCFMh(gfe, resin, fu)

        pes = Material()
        pes.set_props_iso(E1=self.ET,
                          nu12=self.nuT,
                          rho=rho_T)

        # determine mass fractions
        lam_fracs = np.zeros_like(self.lam_angles)
        abs_lam_angles = abs(self.lam_angles)
        idx_layer_45 = np.where(abs_lam_angles == 45.)[0]
        no_layer_45 = len(idx_layer_45)
        idx_layer_0 = np.where(abs_lam_angles == 0.)[0]
        no_layer_0 = len(idx_layer_0)
        idx_layer_90 = np.where(abs_lam_angles == 90.)[0]
        no_layer_90 = len(idx_layer_90)

        if not no_layer_0:
            frac_0 = 0
        if not no_layer_90:
            frac_90 = 0
        if not no_layer_45:
            frac_90 = 1 - frac_0
        else:
            frac_45 = 1 - frac_0 - frac_90

        if no_layer_0:
            lam_fracs[idx_layer_0] = frac_0 / no_layer_0
        if no_layer_45:
            lam_fracs[idx_layer_45] = frac_45 / no_layer_45
        if no_layer_90:
            lam_fracs[idx_layer_90] = frac_90 / no_layer_90

        mA_Fs = lam_fracs * self.mA_F

        #comp.init_matrix_area_dens(mA_Fs, self.psi_M)

        comp.make_laminate(
            mA_Fs, pes, mA_T, self.psi_SF, self.rho_S, self.psi_M)

        comp.lamina_properties(comp.fvf)

        comp.laminate_properties(thicknesses=comp.t_Fs,
                                 angles=self.lam_angles,
                                 stitch=comp.s,
                                 t_T=comp.t_T)

        comp.m.s11_t = self.RM
        # [eps_x, eps_y, eps_x, gamma_yz, gamma_xz, gamma_xy]
        elaminates = np.array([[self.e11_t, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [self.e11_c, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, self.e22_t, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, self.g12]])
        ncase = len(elaminates)
        nlay = len(lam_fracs) + 1  # plus seizing
        #sMs = np.zeros((ncase, nlay))
        sMes = np.zeros((ncase, nlay))
        eMs = np.zeros((ncase, nlay))
        sMemax = np.zeros((ncase))
        for caseid in range(ncase):
            sMes[caseid, :], eMs[caseid, :] = comp.recover_laminate_stresses(
                elaminates[caseid, :])
            sMemax[caseid] = np.max(sMes[caseid, :])

        # objective function
        E1_r = abs(comp.pl.laminate.e1 - self.E1_target) / self.E1_target
        E2_r = abs(comp.pl.laminate.e2 - self.E2_target) / self.E2_target
        G12_r = abs(comp.pl.laminate.g12 - self.G12_target) / self.G12_target

        sum_target_r = E1_r + E2_r + G12_r
        std_target_r = np.std(
            np.array([E1_r, E2_r, G12_r]))

        #EM_r = abs(params['EM'] - self.EM_ref) / self.EM_ref
        #EF_r = abs(params['EF'] - self.EF_ref) / self.EF_ref
        #nuM_r = abs(params['nuM'] - self.nuM_ref) / self.nuM_ref
        #nuF_r = abs(params['nuF'] - self.nuF_ref) / self.nuF_ref

        #sum_init_r = EM_r + EF_r + nuM_r + nuF_r
        #std_init_r = np.std(np.array([EM_r, EF_r, nuM_r, nuF_r]))

        #w_target = self.w_target
        #w_init = 1 - w_target
        # master_r = w_target * \
        #    (sum_target_r + std_target_r) + w_init * (sum_init_r + std_init_r)

        unknowns['rho_G'] = comp.rho_G
        unknowns['rho_M'] = comp.rho_M
        #unknowns['xi_M'] = comp.xi_M
        unknowns['fvf'] = comp.fvf
        unknowns['E1_r'] = E1_r
        unknowns['E2_r'] = E2_r
        unknowns['G12_r'] = G12_r
        unknowns['master_r'] = sum_target_r + std_target_r

        unknowns['t_f'] = comp.t_F
        unknowns['t_T'] = comp.t_T
        unknowns['frac_thicknesses'][:len(lam_fracs)] = lam_fracs
        unknowns['mA_Fs'][:len(mA_Fs)] = mA_Fs
        unknowns['mA_M'] = comp.mA_M
        unknowns['mA_S'] = comp.mA_S
        unknowns['mA_G'] = comp.mA_G
        unknowns['mA_tot'] = comp.mA_tot

        unknowns['sMemax'] = sMemax

    def solve_nonlinear_deprecated(self, params, unknowns, resids):

        fu = params['fu']  # ondulation correction factor
        fkor = params['fkor']  # porosity correction factor
        rho_Gl = params['rho_Gl']  # fiber density
        rho_R = params['rho_R']  # resin density
        EF = params['EF']  # Young's modulus fiber
        EM = params['EM']  # Young's modulus matrix
        nuF = params['nuF']  # poisson's ratio fiber
        nuM = params['nuM']  # poisson's ratio matrix
        frac_0 = params['frac_0']
        frac_90 = params['frac_90']
        mA_T = params['mA_T']  # sewing thread grammage
        rho_T = params['rho_T']  # sewing thread density
        '''
        # UD EGL1600
        # psi_M = 0.29  # matrix mass fraction
        # t_l = 1.243E-03  # thickness laminate
        # rhoA_l = 2.4  # kg/m**2
        # rhoA_f = 1.6  # kg/m**2

        rho_tot = self.mA_tot / self.thickness  # total density

        psi_Gl = self.mA_F / self.mA_tot  # mass fraction fiber+sizing
        # fvf = rho_tot * psi_Gl / rho_Gl  # fiber volume fraction

        psi_T = 1 - psi_Gl - self.psi_M  # mass fraction sewing thread

        psi_f = psi_Gl + psi_T  # mass fraction fabric

        mA_F = self.mA_F + mA_T  # grammage sizing+fiber

        rho_G = psi_f / (psi_Gl / rho_Gl + psi_T / rho_T)  # density fabric

        fvf = mA_F * fkor / self.thickness / rho_G  # fiber volume fraction

        rho_M = rho_tot * self.psi_M / (1 - fvf)  # matrix density
        '''
        resin = Material()
        resin.set_props_iso(E1=EM,
                            nu12=nuM,
                            rho=rho_R)

        gfe = Material()
        gfe.set_props_iso(E1=EF,
                          nu12=nuF,
                          rho=rho_Gl)
        '''
        # ondulation correction for stiffness and poisson's ratio
        # according to Krimmer 2014, eq. 3.1
        gfe.E1 = fu * gfe.E1
        gfe.nu12 = gfe.nu13 = fu * gfe.nu12
        '''
        comp = CompositeCFMh(gfe, resin, fu)

        comp.init_fvf_measurements(
            self.mA_tot, self.mA_F, mA_T, self.psi_M, rho_Gl, rho_T,
            self.thickness, rho_R, fkor, self.rho_S, self.psi_SF)

        comp.lamina_properties(comp.fvf)

        # determ_tine thickness fractions
        lam_fracs = np.zeros_like(self.lam_angles)
        abs_lam_angles = abs(self.lam_angles)
        idx_layer_45 = np.where(abs_lam_angles == 45.)[0]
        no_layer_45 = len(idx_layer_45)
        idx_layer_0 = np.where(abs_lam_angles == 0.)[0]
        no_layer_0 = len(idx_layer_0)
        idx_layer_90 = np.where(abs_lam_angles == 90.)[0]
        no_layer_90 = len(idx_layer_90)

        if not no_layer_0:
            frac_0 = 0
        if not no_layer_90:
            frac_90 = 0
        if not no_layer_45:
            frac_90 = 1 - frac_0
        else:
            frac_45 = 1 - frac_0 - frac_90

        if no_layer_0:
            lam_fracs[idx_layer_0] = frac_0 / no_layer_0
        if no_layer_45:
            lam_fracs[idx_layer_45] = frac_45 / no_layer_45
        if no_layer_90:
            lam_fracs[idx_layer_90] = frac_90 / no_layer_90

        t_f = comp.t_f
        t_T = comp.t_T
        frac_thicknesses = lam_fracs * (t_f - t_T)

        # determ_tine grammage per layer
        grammage = np.zeros_like(frac_thicknesses)
        for i, ti in enumerate(frac_thicknesses):
            grammage[i] = ti * comp.rho_F * comp.fvf

        self.pes = Material()
        self.pes.set_props_iso(E1=self.ET,
                               nu12=self.nuT,
                               rho=rho_T)

        comp.laminate_properties(
            frac_thicknesses, self.lam_angles, self.pes, t_T)

        comp.m.s11_t = self.RM
        # [eps_x, eps_y, eps_x, gamma_yz, gamma_xz, gamma_xy]
        elaminates = np.array([[self.e11_t, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [self.e11_c, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, self.e22_t, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, self.g12]])
        ncase = len(elaminates)
        nlay = len(lam_fracs) + 1  # plus seizing
        #sMs = np.zeros((ncase, nlay))
        sMes = np.zeros((ncase, nlay))
        eMs = np.zeros((ncase, nlay))
        sMemax = np.zeros((ncase))
        for caseid in range(ncase):
            sMes[caseid, :], eMs[caseid, :] = comp.recover_laminate_stresses(
                elaminates[caseid, :])
            sMemax[caseid] = np.max(sMes[caseid, :])

        # objective function
        E1_r = abs(comp.pl.laminate.e1 - self.E1_target) / self.E1_target
        E2_r = abs(comp.pl.laminate.e2 - self.E2_target) / self.E2_target
        G12_r = abs(comp.pl.laminate.g12 - self.G12_target) / self.G12_target

        sum_target_r = E1_r + E2_r + G12_r
        std_target_r = np.std(
            np.array([E1_r, E2_r, G12_r]))

        EM_r = abs(params['EM'] - self.EM_ref) / self.EM_ref
        EF_r = abs(params['EF'] - self.EF_ref) / self.EF_ref
        nuM_r = abs(params['nuM'] - self.nuM_ref) / self.nuM_ref
        nuF_r = abs(params['nuF'] - self.nuF_ref) / self.nuF_ref

        sum_init_r = EM_r + EF_r + nuM_r + nuF_r
        std_init_r = np.std(np.array([EM_r, EF_r, nuM_r, nuF_r]))

        w_target = self.w_target
        w_init = 1 - w_target
        master_r = w_target * \
            (sum_target_r + std_target_r) + w_init * (sum_init_r + std_init_r)

        unknowns['rho_G'] = comp.rho_G
        unknowns['rho_M'] = comp.rho_M
        unknowns['xi_M'] = comp.xi_M
        unknowns['fvf'] = comp.fvf
        unknowns['E1_r'] = E1_r
        unknowns['E2_r'] = E2_r
        unknowns['G12_r'] = G12_r
        unknowns['master_r'] = master_r

        unknowns['t_f'] = t_f
        unknowns['t_T'] = t_T
        unknowns['frac_thicknesses'][:len(frac_thicknesses)] = frac_thicknesses
        unknowns['grammage'][:len(grammage)] = grammage
        unknowns['mA_M'] = comp.mA_M
        unknowns['mA_S'] = comp.mA_S
        unknowns['mA_G'] = comp.mA_G

        unknowns['sMemax'] = sMemax


def optimize_material(cfg):

    top = Problem(root=Group())
    top.root.add('material', CompositeCFMhComponent(cfg), promotes=['*'])

    #top.root.add('fu_c', IndepVarComp('fu', 1.0), promotes=['*'])
    #top.root.add('fkor_c', IndepVarComp('fkor', 1.0), promotes=['*'])
    #top.root.add('rho_Gl_c', IndepVarComp('rho_Gl', 2500.), promotes=['*'])
    #top.root.add('rho_R_c', IndepVarComp('rho_R', 1150.), promotes=['*'])
    # 81.383E+09
    #top.root.add('EF_c', IndepVarComp('EF', cfg['EF_ref']), promotes=['*'])
    #top.root.add('EM_c', IndepVarComp('EM', cfg['EM_ref']), promotes=['*'])
    #top.root.add('nuF_c', IndepVarComp('nuF', cfg['nuF_ref']), promotes=['*'])
    #top.root.add('nuM_c', IndepVarComp('nuM', cfg['nuM_ref']), promotes=['*'])
    top.root.add('frac_0_c', IndepVarComp('frac_0', 0.25), promotes=['*'])
    top.root.add('frac_90_c', IndepVarComp('frac_90', 0.25), promotes=['*'])
    top.root.add('mA_T_c', IndepVarComp('mA_T', 0.006), promotes=['*'])
    #top.root.add('fvf_c', IndepVarComp('fvf', 0.55), promotes=['*'])
    #top.root.add('rho_T_c', IndepVarComp('rho_T', 1370.), promotes=['*'])

    top.driver = pyOptSparseDriver()
    top.driver.options['optimizer'] = 'NSGA2'
    top.driver.opt_settings['PopSize'] = cfg['opt_settings']['PopSize']  # 100
    top.driver.opt_settings['maxGen'] = cfg['opt_settings']['maxGen']  # 150
    top.driver.opt_settings['PrintOut'] = 1
    '''
    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'COBYLA'  # 'SLSQP'
    # maximum number of solver iterations
    top.driver.options['maxiter'] = 200
    top.driver.options['tol'] = 1.0E-6
    '''
    #top.driver.add_desvar('fu', lower=cfg['fu_lower'], upper=cfg['fu_upper'])
    # top.driver.add_desvar(
    #    'fkor', lower=cfg['fkor_lower'], upper=cfg['fkor_upper'])
    # top.driver.add_desvar(
    #    'rho_Gl', lower=cfg['rho_Gl_lower'], upper=cfg['rho_Gl_upper'])
    # top.driver.add_desvar(
    #    'rho_R', lower=cfg['rho_R_lower'], upper=cfg['rho_R_upper'])
    #top.driver.add_desvar('EF', lower=cfg['EF_lower'], upper=cfg['EF_upper'])
    #top.driver.add_desvar('EM', lower=cfg['EM_lower'], upper=cfg['EM_upper'])
    # top.driver.add_desvar('nuF', lower=cfg['nuF_lower'],
    #                      upper=cfg['nuF_upper'])
    # top.driver.add_desvar('nuM', lower=cfg['nuM_lower'],
    #                      upper=cfg['nuM_upper'])
    top.driver.add_desvar('frac_0', lower=cfg['frac_0_lower'],
                          upper=cfg['frac_0_upper'])
    top.driver.add_desvar('frac_90', lower=cfg['frac_90_lower'],
                          upper=cfg['frac_90_upper'])
    top.driver.add_desvar('mA_T', lower=cfg['mA_T_lower'],
                          upper=cfg['mA_T_upper'])
    # top.driver.add_desvar('fvf', lower=cfg['fvf_lower'],
    #                      upper=cfg['fvf_upper'])
    # top.driver.add_desvar('rho_T', lower=cfg['rho_T_lower'],
    #                      upper=cfg['rho_T_upper'])

    top.driver.add_objective('master_r')

    # add the recorder
    recorder = SqliteRecorder(cfg['filename_db'])
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    top.driver.add_recorder(recorder)

    top.setup()
    top.run()


def post_optimization(cfg):
    db = sqlitedict.SqliteDict(cfg['filename_db'], 'iterations')
    #data = db['rank0:NSGA2|1']
    itrs = len(db.keys())
    print itrs
    #u = data['Unknowns']
    #p = data['Parameters']
    PopSize = 100
    post_range = range(itrs - 1 * PopSize, itrs)
    E1_r = np.zeros((len(post_range)))
    E2_r = np.zeros((len(post_range)))
    G12_r = np.zeros((len(post_range)))
    #fu = np.zeros((len(post_range)))
    #fkor = np.zeros((len(post_range)))
    #rho_Gl = np.zeros((len(post_range)))
    #rho_R = np.zeros((len(post_range)))
    #xi_M = np.zeros((len(post_range)))
    #rho_M = np.zeros((len(post_range)))
    fvf = np.zeros((len(post_range)))
    #EF = np.zeros((len(post_range)))
    #EM = np.zeros((len(post_range)))
    #nuF = np.zeros((len(post_range)))
    #nuM = np.zeros((len(post_range)))
    frac_0 = np.zeros((len(post_range)))
    frac_90 = np.zeros((len(post_range)))
    mA_T = np.zeros((len(post_range)))
    #rho_T = np.zeros((len(post_range)))
    rho_G = np.zeros((len(post_range)))
    #sum_r = np.zeros((itrs))
    #std_r = np.zeros((itrs))
    master_r = np.zeros((len(post_range)))
    t_T = np.zeros((len(post_range)))
    t_f = np.zeros((len(post_range)))
    frac_thicknesses = np.zeros((len(post_range), 4))
    mA_Fs = np.zeros((len(post_range), 4))
    mA_M = np.zeros((len(post_range)))
    mA_S = np.zeros((len(post_range)))
    mA_G = np.zeros((len(post_range)))
    mA_tot = np.zeros((len(post_range)))

    sMemax = np.zeros((len(post_range), 4))

    for i, it in enumerate(post_range):
        E1_r[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['E1_r']
        E2_r[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['E2_r']
        G12_r[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['G12_r']
        #fu[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['fu']
        #fkor[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['fkor']
        #rho_Gl[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['rho_Gl']
        #rho_R[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['rho_R']
        #xi_M[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['xi_M']
        #rho_M[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['rho_M']
        fvf[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['fvf']
        #EF[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['EF']
        #EM[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['EM']
        #nuF[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['nuF']
        #nuM[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['nuM']
        frac_0[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['frac_0']
        frac_90[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['frac_90']
        mA_T[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['mA_T']
        #rho_T[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['rho_T']
        rho_G[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['rho_G']
        #sum_r[i] = E1_r[i] + E2_r[i] + G12_r[i]
        #std_r[i] = np.std(np.array([E1_r[i], E2_r[i], G12_r[i]]))
        master_r[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['master_r']
        t_f[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['t_f']
        t_T[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['t_T']
        frac_thicknesses[i, :] = db['rank0:NSGA2|%s' %
                                    (it + 1)]['Unknowns']['frac_thicknesses']
        mA_Fs[i, :] = db['rank0:NSGA2|%s' %
                         (it + 1)]['Unknowns']['mA_Fs']
        mA_M[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['mA_M']
        mA_S[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['mA_S']
        mA_G[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['mA_G']
        mA_tot[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['mA_tot']

        sMemax[i] = db['rank0:NSGA2|%s' % (it + 1)]['Unknowns']['sMemax']

    # print E1_r[:popSize]
    # print E2_r[:popSize]
    # print G12_r[:popSize]

    #midxs = np.argpartition(master_r, 5)
    midx = master_r.argmin()
    print midx
    print master_r[midx]
    print E1_r[midx]
    print E2_r[midx]
    print G12_r[midx]
    # print midxs
    # print master_r[midxs]
    # print E1_r[midxs]
    # print E2_r[midxs]
    # print G12_r[midxs]
    # find minium of last generation

    #idx = E1_r[-popSize:].argmin()
    #E2_rm_tin = E2_r[-100:].argmin()
    #G12_rm_tin = E2_r[-100:].argmin()\

    print 'E1_r = %.3f' % E1_r[midx]
    print 'E2_r = %.3f' % E2_r[midx]
    print 'G12_r = %.3f' % G12_r[midx]

    # print 'fu = %.4f' % fu[midx]
    # print 'fkor = %.4f' % fkor[midx]
    print 'frac_0 = %.3f' % frac_0[midx]
    print 'frac_90 = %.3f' % frac_90[midx]
    # print fu[midxs]
    # print 'rho_G = %.2f' % rho_G[midx]
    # print 'rho_Gl = %.2f' % rho_Gl[midx]
    # print 'rho_T = %.2f' % rho_T[midx]
    # print 'rho_M = %.2f' % rho_M[midx]
    # print 'rho_R = %.2f' % rho_R[midx]
    # print 'xi_M = %.2f' % xi_M[midx]
    print 'mA_T = %.4f' % mA_T[midx]

    print 'fvf = %.3f' % fvf[midx]

    # print rho_Gl[midxs]
    # print 'EF = %.2f' % EF[midx]
    # print EF[midxs]
    # print 'EM = %.2f' % EM[midx]
    # print EM[midxs]
    # print 'nuF = %.3f' % nuF[midx]
    # print nuF[midxs]
    # print 'nuM = %.3f' % nuM[midx]
    # print nuM[midxs]

    #cfg['fu'] = fu[midx]
    #cfg['fkor'] = fkor[midx]
    cfg['frac_0'] = frac_0[midx]
    cfg['frac_90'] = frac_90[midx]
    #cfg['rho_Gl'] = rho_Gl[midx]
    cfg['fvf'] = fvf[midx]
    #cfg['rho_M'] = rho_M[midx]
    #cfg['rho_R'] = rho_R[midx]
    #cfg['xi_M'] = xi_M[midx]
    #cfg['EF'] = EF[midx]
    #cfg['EM'] = EM[midx]
    #cfg['nuF'] = nuF[midx]
    #cfg['nuM'] = nuM[midx]
    cfg['mA_T'] = mA_T[midx]
    #cfg['rho_T'] = rho_T[midx]
    #cfg['rho_G'] = rho_G[midx]

    cfg['t_f'] = t_f[midx]
    cfg['t_T'] = t_T[midx]
    cfg['frac_thicknesses'] = frac_thicknesses[midx]
    cfg['mA_Fs'] = mA_Fs[midx]
    cfg['mA_M'] = mA_M[midx]
    cfg['mA_S'] = mA_S[midx]
    cfg['mA_G'] = mA_G[midx]
    cfg['mA_tot'] = mA_tot[midx]

    cfg['sMemax'] = sMemax[midx]

    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(E1_r, E2_r, 'b.')
    #plt.plot(E1_r[:popSize], E2_r[:popSize], 'ko')
    #plt.plot(E1_r[popSize:2 * popSize], E2_r[popSize:2 * popSize], 'go')
    #plt.plot(E1_r[-popSize:], E2_r[-popSize:], 'ro')
    plt.plot(E1_r[midx], E2_r[midx], 'mo')
    #plt.xlim(0.0, 0.05)
    #plt.ylim(0.0, 0.05)
    plt.savefig('pareto_E1_E2.png', dpi=200)

    plt.figure()
    plt.plot(E1_r, G12_r, 'b.')
    #plt.plot(E1_r[:popSize], G12_r[:popSize], 'ko')
    #plt.plot(E1_r[popSize:2 * popSize], G12_r[popSize:2 * popSize], 'go')
    #plt.plot(E1_r[-popSize:], G12_r[-popSize:], 'ro')
    plt.plot(E1_r[midx], G12_r[midx], 'mo')
    #plt.xlim(0.0, 0.05)
    #plt.ylim(0.0, 0.05)
    plt.savefig('pareto_E1_G12.png', dpi=200)

    plt.figure()
    plt.plot(E2_r, G12_r, 'b.')
    #plt.plot(E2_r[:popSize], G12_r[:popSize], 'ko')
    #plt.plot(E2_r[popSize:2 * popSize], G12_r[popSize:2 * popSize], 'go')
    #plt.plot(E2_r[-popSize:], G12_r[-popSize:], 'ro')
    plt.plot(E2_r[midx], G12_r[midx], 'mo')
    #plt.xlim(0.0, 0.05)
    #plt.ylim(0.0, 0.05)
    plt.savefig('pareto_E2_G12.png', dpi=200)

    '''
    plt.figure()
    for i in range(itrs):
        plt.plot(i, db['rank0:NSGA2|%s' % (i + 1)]['Unknowns']['E1_r'], 'b.')
        plt.plot(i, db['rank0:NSGA2|%s' % (i + 1)]['Unknowns']['E2_r'], 'g.')
        plt.plot(i, db['rank0:NSGA2|%s' % (i + 1)]['Unknowns']['G12_r'], 'r.')
    '''
    #plt.savefig('residuals.png', dpi=200)
    # plt.show()


def vary_fvf(cfg, thickness):
    ''' determ_tine properties by varying resin content through lam. thickness
    '''

    resin = Material()
    resin.set_props_iso(E1=cfg['EM'],
                        nu12=cfg['nuM'],
                        rho=cfg['rho_R'])
    resin.s11_t = cfg['RM']

    gfe = Material()
    gfe.set_props_iso(E1=cfg['EF'],
                      nu12=cfg['nuF'],
                      rho=cfg['rho_Gl'])

    compv = CompositeCFMh(gfe, resin, cfg['fu'])

    # determ_tine new tot mass and matrix mass fraction
    mA_Madd = cfg['rho_R'] * (thickness / cfg['fkor'] - cfg['t_f'])
    mA_tot_var = cfg['mA_tot'] + mA_Madd
    mA_M_orig = cfg['mA_tot'] - cfg['mA_F'] - cfg['mA_T']
    psi_M_var = (mA_M_orig + mA_Madd) / mA_tot_var

    #psi_Gl_orig = cfg['mA_F'] / cfg['mA_tot']
    #psi_T_orig = cfg['mA_T'] / cfg['mA_tot']
    #psi_f_orig = (cfg['mA_F'] + cfg['mA_T']) / cfg['mA_tot']
    #psi_M_orig = cfg['psi_M']
    #rest = 1 - np.sum([psi_Gl_orig, psi_T_orig, psi_M_orig])
    #rm_tf = rest / cfg['mA_tot']

    #psi_Gl_a = 1- psi_M_orig - psi_T_orig
    #mA_F_a = psi_Gl_a * cfg['mA_tot']

    #psi_M_kor = 1 - psi_Gl_orig - psi_T_orig

    compv.init_fvf_measurements(mA_tot_var,
                                cfg['mA_F'],
                                cfg['mA_T'],
                                psi_M_var,
                                cfg['rho_Gl'],
                                cfg['rho_T'],
                                thickness,
                                cfg['rho_R'],
                                cfg['fkor'],
                                cfg['rho_S'],
                                cfg['psi_SF'])

    compv.lamina_properties(compv.fvf)

    # determ_tine thickness fractions
    frac_0 = cfg['frac_0']
    frac_90 = cfg['frac_90']

    lam_fracs = np.zeros_like(cfg['lam_angles'])
    abs_lam_angles = abs(cfg['lam_angles'])
    idx_layer_45 = np.where(abs_lam_angles == 45.)[0]
    no_layer_45 = len(idx_layer_45)
    idx_layer_0 = np.where(abs_lam_angles == 0.)[0]
    no_layer_0 = len(idx_layer_0)
    idx_layer_90 = np.where(abs_lam_angles == 90.)[0]
    no_layer_90 = len(idx_layer_90)

    if not no_layer_0:
        frac_0 = 0
    if not no_layer_90:
        frac_90 = 0
    if not no_layer_45:
        frac_90 = 1 - frac_0
    else:
        frac_45 = 1 - frac_0 - frac_90

    if no_layer_0:
        lam_fracs[idx_layer_0] = frac_0 / no_layer_0
    if no_layer_45:
        lam_fracs[idx_layer_45] = frac_45 / no_layer_45
    if no_layer_90:
        lam_fracs[idx_layer_90] = frac_90 / no_layer_90

    frac_thicknesses = lam_fracs * thickness

    compv.laminate_properties(frac_thicknesses, cfg['lam_angles'])

    # determ_tine grammage per layer
    grammage = np.zeros_like(frac_thicknesses)
    for i, ti in enumerate(frac_thicknesses):
        grammage[i] = ti * compv.rho_F * compv.fvf
        #grammage[i] = ti * cfg['rho_Gl'] * compv.fvf

    print np.sum(grammage)
    #
    # [eps_x, eps_y, eps_x, gamma_yz, gamma_xz, gamma_xy]
    elaminates = np.array([[cfg['e11_t'], 0.0, 0.0, 0.0, 0.0, 0.0],
                           [-cfg['e11_c'], 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, cfg['e22_t'], 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, cfg['g12']]])
    ncase = len(elaminates)
    nlay = len(lam_fracs) + 1  # p[us seizing
    eMs = np.zeros((ncase, nlay))
    for caseid in range(ncase):
        eMs[caseid, :] = compv.recover_laminate_stresses(elaminates[caseid, :])
        print eMs[caseid, :]

    cfgn = copy.deepcopy(cfg)

    cfgn['psi_M'] = psi_M_var
    cfgn['mA_tot'] = mA_tot_var
    cfgn['thickness'] = thickness
    cfgn['rho_G'] = compv.rho_G
    cfgn['rho_M'] = compv.rho_M
    cfgn['fkor'] = compv.fkor
    cfgn['fvf'] = compv.fvf
    cfgn['E1_target'] = compv.pl.laminate.e1
    cfgn['E2_target'] = compv.pl.laminate.e2
    cfgn['G12_target'] = compv.pl.laminate.g12

    return cfgn


def write_result_file(cfg):
    headers = [r'$\gls{E}_{\gls{F}}$',
               r'$\gls{nu}_{\gls{F}}$',
               r'$\gls{E}_{\gls{M}}$',
               r'$\gls{nu}_{\gls{M}}$',
               r'$\gls{phi}$',
               r'$\gls{f}_{\gls{kor}}$',
               r'$\gls{f}_{\gls{u}}$']
    units = [r'\si{\giga \pascal}',
             r'\si{}',
             r'\si{\giga \pascal}',
             r'\si{}',
             r'\si{\percent}',
             r'\si{}',
             r'\si{}',
             ]
    floatfmt = ['g',
                '3.4f',
                '3.4f',
                '3.4f',
                '3.4f',
                '3.2f',
                '3.4f',
                '3.4f',
                ]
    header = ''.join(headers)
    cols = len(headers)
    lines = len(cfg.keys()) + 1
    res_array = np.zeros((lines, cols))
    for mi, m in enumerate(cfg.iterkeys()):
        try:
            res_array[mi, 0] = cfg[m]['EF'] * 1.0E-09
            res_array[mi, 1] = cfg[m]['nuF']
            res_array[mi, 2] = cfg[m]['EM'] * 1.0E-09
            res_array[mi, 3] = cfg[m]['nuM']
            res_array[mi, 4] = cfg[m]['fvf'] * 1.0E+02
            res_array[mi, 5] = cfg[m]['fkor']
            res_array[mi, 6] = cfg[m]['fu']
        except:
            pass
    # add reference
    res_array[-1, 0] = cfg[m]['EF_ref'] * 1.0E-09
    res_array[-1, 1] = cfg[m]['nuF_ref']
    res_array[-1, 2] = cfg[m]['EM_ref'] * 1.0E-09
    res_array[-1, 3] = cfg[m]['nuM_ref']
    res_array[-1, 4] = np.nan
    res_array[-1, 5] = np.nan
    res_array[-1, 6] = np.nan

    table_list = res_array.tolist()
    for k, table_line in zip(cfg.iterkeys(), table_list):
        table_line.insert(0, k)
    # add reference
    table_list[-1].insert(0, r'\textcite{krimmer2014micro}')
    for idx, item in enumerate(table_list[-1]):
        if 'nan' in str(item):
            table_list[-1][idx] = None

    headers.insert(0, 'Property')
    units.insert(0, 'Unit')

    print tabulate.tabulate(table_list, headers=headers, headers2=units)
    np.savetxt('material_opt.dat', res_array, header=header)
    report_table(table_list, headers, units, floatfmt,
                 filename=os.path.join(
                     report_path, 'material_properties_inverse'),
                 caption='SSP 34 inverse material properties.',
                 label='tab:mat_ssp_inv')


def write_layup_file(cfg, cfgr):
    # thickness
    # 'fkor',
    headers = ['t_tot', 't_f', 't_T', 't_0', 't_90', 't_45', 't_-45']
    floatfmt = 'g'
    header = ''.join(headers)
    cols = 6
    lines = len(cfg.keys())
    res_array = np.zeros((lines, cols))
    for mi, m in enumerate(cfg.iterkeys()):
        try:
            res_array[mi, 0] = cfg[m]['t_f'] + cfg[m]['t_T']
            res_array[mi, 1] = cfg[m]['t_f']
            res_array[mi, 2] = cfg[m]['t_T']
            for i, t in enumerate(cfg[m]['frac_thicknesses']):
                res_array[mi, 3 + i] = t
        except:
            pass

    # to mm
    res_array = res_array * 1.0E+03

    res_list = res_array.tolist()
    for k, res_list_line in zip(cfg.iterkeys(), res_list):
        res_list_line.insert(0, k)
    headers.insert(0, 'material')

    print tabulate.tabulate(res_list, headers=headers)
    print tabulate.tabulate(res_list, headers=headers, tablefmt='latex', floatfmt=floatfmt)
    np.savetxt('material_opt.dat', res_array, header=header)

    # grammage
    headers = [r'$\gls{mA}_{\gls{tot}}$',
               r'$\gls{mA}_{\gls{M}}$',
               r'$\gls{mA}_{\gls{f_us}}$',
               r'$\gls{mA}_{\gls{F}}$',
               r'$\gls{mA}_{\gls{S}}$',
               r'$\gls{mA}_{\gls{T}}$',
               r'$\gls{mA}_{\gls{zero}}$',
               r'$\gls{mA}_{\gls{ninety}}$',
               r'$\gls{mA}_{\gls{plusff}}$',
               r'$\gls{mA}_{\gls{minff}}$', ]
    units = [r'\si{\gram \meter^{-2}}'] * len(headers)
    floatfmt = '3.0f'
    header = ''.join(headers)
    cols = len(headers)
    lines = len(cfg.keys()) + len(cfgr.keys())
    res_array = np.zeros((lines, cols))
    for mi, m in enumerate(cfg.iterkeys()):
        try:
            res_array[mi, 0] = cfg[m]['mA_tot']
            res_array[mi, 1] = cfg[m]['mA_M']
            res_array[mi, 2] = cfg[m]['mA_G']
            res_array[mi, 3] = cfg[m]['mA_F']
            res_array[mi, 4] = cfg[m]['mA_S']
            res_array[mi, 5] = cfg[m]['mA_T']
            for i, t in enumerate(cfg[m]['mA_Fs']):
                res_array[mi, 6 + i] = t
        except:
            pass
    mim = len(cfg.keys())
    # add reference fabrics
    for mi, m in enumerate(cfgr.iterkeys()):
        res_array[mi + mim, 0] = np.nan
        res_array[mi + mim, 1] = np.nan
        res_array[mi + mim, 2] = cfgr[m]['mA_G']
        res_array[mi + mim, 3] = cfgr[m]['mA_F']
        res_array[mi + mim, 4] = np.nan
        res_array[mi + mim, 5] = cfgr[m]['mA_T']
        for i, t in enumerate(cfgr[m]['mA_Fs']):
            res_array[mi + mim, 6 + i] = t

    res_array = res_array * 1.0E+03

    table_list = res_array.tolist()
    keys = []
    for k in cfg.keys():
        keys.append(k)
    for k in cfgr.keys():
        keys.append(k + cfgr[k]['cite'])
    for k, table_list_line in zip(keys, table_list):
        table_list_line.insert(0, k)
        for idx, item in enumerate(table_list_line):
            if 'nan' in str(item):
                table_list_line[idx] = None

    headers.insert(0, 'Property')
    units.insert(0, 'Unit')

    print tabulate.tabulate(table_list, headers=headers)
    np.savetxt('material_grammage.dat', res_array, header=header)

    report_table(table_list, headers, units, floatfmt,
                 filename=os.path.join(report_path, 'material_grammage'),
                 caption='SSP 34 inverse determined material area densities.',
                 label='tab:mat_ssp_gram')


def write_property_file(cfg):
    headers = [r'$\gls{E}_{\gls{x_us}}$',
               r'$\gls{E}_{\gls{y_us}}$',
               r'$\gls{G}_{\gls{x_us}\gls{y_us}}$',
               r'$\gls{mA}_{\gls{tot}}$',
               r'$\gls{mA}_{\gls{F}}$',
               r'$\gls{psi}_{\gls{M}}$\footnote{Not used within calculation.}',
               r'$\gls{t}_{\gls{tot}}$',
               'Construction']
    units = [r'\si{\giga \pascal}',
             r'\si{\giga \pascal}',
             r'\si{\giga \pascal}',
             r'\si{\kg \meter^{-2}}',
             r'\si{\kg \meter^{-2}}',
             r'\si{\percent}',
             r'\si{\milli \meter}',
             r' ']
    floatfmt = ['g',
                '3.2f',
                '3.2f',
                '3.2f',
                '3.3f',
                '3.3f',
                '3.0f',
                '3.3f',
                'g'
                ]
    header = ''.join(headers)
    unit = ''.join(units)
    cols = len(headers) - 1
    lines = len(cfg.keys())
    prop_array = np.zeros((lines, cols))

    for mi, m in enumerate(cfg.iterkeys()):
        try:
            prop_array[mi, 0] = cfg[m]['E1_target'] * 1.0E-09
            prop_array[mi, 1] = cfg[m]['E2_target'] * 1.0E-09
            prop_array[mi, 2] = cfg[m]['G12_target'] * 1.0E-09
            prop_array[mi, 3] = cfg[m]['mA_tot']
            prop_array[mi, 4] = cfg[m]['mA_F']
            prop_array[mi, 5] = cfg[m]['psi_M'] * 1.0E+02
            prop_array[mi, 6] = (cfg[m]['t_f'] + cfg[m]['t_T']) * 1.0E+03
        except:
            pass

    lam_angles_list = []
    for m in cfg.iterkeys():
        lam_angles_list.append(np.array_str(cfg[m]['lam_angles'], precision=1))

    table_list = prop_array.tolist()
    for k, list_line, lam_angles in zip(cfg.iterkeys(), table_list, lam_angles_list):
        list_line.insert(0, k)
        list_line.append(lam_angles)
    headers.insert(0, 'Property')
    units.insert(0, 'Unit')

    print tabulate.tabulate(table_list, headers=headers, headers2=units, floatfmt=floatfmt)
    np.savetxt('material_properties.dat', prop_array, header=header)

    report_table(table_list, headers, units, floatfmt,
                 filename=os.path.join(report_path, 'material_properties'),
                 caption='SSP 34 smeared material properties.',
                 label='tab:mat_ssp')

    headers = [r'$\gls{eps}_{\gls{x_us}}^{\gls{t_ss}}$',
               r'$\gls{eps}_{\gls{x_us}}^{\gls{c_ss}}$',
               r'$\gls{eps}_{\gls{y_us}}^{\gls{t_ss}}$',
               r'$\gls{gamma}_{\gls{x_us}\gls{y_us}}$',
               ]
    units = [r'\si{\micro}',
             r'\si{\micro}',
             r'\si{\micro}',
             r'\si{\micro}']
    floatfmt = '3.0f'
    header = ''.join(headers)
    unit = ''.join(units)
    cols = len(headers)
    lines = len(cfg.keys())
    prop_array = np.zeros((lines, cols))

    for mi, m in enumerate(cfg.iterkeys()):
        prop_array[mi, 0] = cfg[m]['e11_t'] * 1.0E+06
        prop_array[mi, 1] = cfg[m]['e11_c'] * 1.0E+06
        prop_array[mi, 2] = cfg[m]['e22_t'] * 1.0E+06
        prop_array[mi, 3] = cfg[m]['g12'] * 1.0E+06

    table_list = prop_array.tolist()
    for k, list_line in zip(cfg.iterkeys(), table_list):
        list_line.insert(0, k)
    headers.insert(0, 'Property')
    units.insert(0, 'Unit')

    print tabulate.tabulate(table_list, headers=headers, headers2=units, floatfmt=floatfmt)
    np.savetxt('material_strains.dat', prop_array, header=header)

    report_table(table_list, headers, units, floatfmt,
                 filename=os.path.join(report_path, 'material_strains'),
                 caption='SSP 34 material allowable strains.',
                 label='tab:mat_ssp_strains')


def write_constants_file(cfgc):

    doc = Document(os.path.join(report_path, 'material_constants_list'))
    with doc.create(Itemize()) as itemize:
        itemize.add_item(
            NoEscape(r"""$\gls{E}_{\gls{T}} = \SI{%3.2f}{\giga \pascal}$""" % (cfgc['ET'] * 1E-09)))
        itemize.add_item(
            NoEscape(r"""$\gls{nu}_{\gls{T}} = \SI{%3.2f}{}$""" % cfgc['nuT']))
        itemize.add_item(
            NoEscape(r"""$\gls{rho}_{\gls{T}} = \SI{%3.0f}{\kg \meter^{-3}}$""" % cfgc['rho_T']))
        itemize.add_item(
            NoEscape(r"""$\gls{rho}_{\gls{R}} = \SI{%3.0f}{\kg \meter^{-3}}$""" % cfgc['rho_R']))
        itemize.add_item(
            NoEscape(r"""$\gls{rho}_{\gls{Gl}} = \SI{%3.0f}{\kg \meter^{-3}}$""" % cfgc['rho_Gl']))
        itemize.add_item(
            NoEscape(r"""$\gls{rho}_{\gls{S}} = \SI{%3.0f}{\kg \meter^{-3}}$""" % cfgc['rho_S']))
        itemize.add_item(
            NoEscape(r"""$\gls{psi}_{\gls{SF}} = \SI{%3.2f}{\percent}$""" % (cfgc['psi_SF'] * 1E+02)))
    doc.generate_tex_content_only()


def write_stress_file(cfg, cfgr2):
    headers = [r'$\gls{R}_{\gls{M}}^{\gls{t_ss}}$',
               r'$\gls{R}_{\gls{M}}^{\gls{c_ss}}$',
               r'$\gls{R}_{\gls{M}}^{\gls{t_ss}}$',
               r'$\gls{R}_{\gls{M}}^{\gls{s_ss}}$']
    units = [r'\si{\mega \pascal}',
             r'\si{\mega \pascal}',
             r'\si{\mega \pascal}',
             r'\si{\mega \pascal}']
    floatfmt = '3.1f'
    header = ''.join(headers)
    unit = ''.join(units)
    cols = len(headers)
    lines = len(cfg.keys()) + 1
    prop_array = np.zeros((lines, cols))

    for mi, m in enumerate(cfg.iterkeys()):
        try:
            prop_array[mi, :] = cfg[m]['sMemax'] * 1.0E-06
        except:
            pass

    # add reference
    m = 'Krimmer'
    prop_array[-1, 0] = cfgr2[m]['rm_t'] * 1.0E-06
    prop_array[-1, 1] = cfgr2[m]['rm_c'] * 1.0E-06
    prop_array[-1, 2] = cfgr2[m]['rm_t'] * 1.0E-06
    prop_array[-1, 3] = cfgr2[m]['rm_s'] * 1.0E-06

    table_list = prop_array.tolist()
    for k, list_line in zip(cfg.iterkeys(), table_list):
        list_line.insert(0, k)
    headers.insert(0, 'Property')
    units.insert(0, 'Unit')

    # add reference
    table_list[-1].insert(0, r'\textcite{krimmer2014micro}')
    for idx, item in enumerate(table_list[-1]):
        if 'nan' in str(item):
            table_list[-1][idx] = None

    print tabulate.tabulate(table_list, headers=headers, headers2=units, floatfmt=floatfmt)
    np.savetxt('material_resists.dat', prop_array, header=header)

    report_table(table_list, headers, units, floatfmt,
                 filename=os.path.join(report_path, 'material_resists'),
                 caption='SSP 34 material matrix resistances.',
                 label='tab:mat_ssp_resists')


def report_table(table_list, headers, units, floatfmt, filename, caption, label):
    latex_tab = tabulate.tabulate(
        table_list,
        headers=headers,
        headers2=units,
        tablefmt='latex_raw_booktabs',
        floatfmt=floatfmt)

    # table to latex
    doc = Document(filename)
    doc.append(NoEscape(r"""
\begin{table}[htbp]
\sffamily
\sansmath
\centering
\caption{%s}
\label{%s}
""" % (caption, label)))
    doc.append(NoEscape(latex_tab))
    doc.append(NoEscape(r"""\end{table}"""))
    doc.generate_tex_content_only()


if __name__ == '__main__':

    report_path = '/home/mrosemeier/iwes-gitlab/bdtprojects/VALDEMOD_report/tables/'
    cfgc = OrderedDict()
    cfgc['ET'] = ET = 1.5E+10
    cfgc['nuT'] = nuT = 0.28
    cfgc['rho_T'] = rho_T = 1370.0  # kg/m**3
    cfgc['rho_R'] = rho_R = 1300  # 1250  # 1150.0  # kg/m**3
    cfgc['rho_Gl'] = rho_Gl = 2600.0  # kg/m**3
    cfgc['rho_S'] = rho_S = 1150.0  # kg/m**3
    cfgc['psi_SF'] = psi_SF = 0.0055

    cfgr = OrderedDict()
    cfgm = cfgr['U-E-1200'] = {}
    cfgm['cite'] = r' \cite{saertexue1200}'
    cfgm['mA_G'] = 1.200  # kg/m**2
    cfgm['mA_F'] = 1.188  # kg/m**2
    cfgm['mA_T'] = 0.012  # kg/m**2
    cfgm['mA_Fs'] = np.array([1.134, 0.054, 0.0, 0.0])

    cfgm = cfgr['X-E-612'] = {}
    cfgm['cite'] = r' \cite{saertexxe612}'
    cfgm['mA_G'] = 0.612  # kg/m**2
    cfgm['mA_F'] = 0.606  # kg/m**2
    cfgm['mA_T'] = 0.006  # kg/m**2
    cfgm['mA_Fs'] = np.array([0.003, 0.003, 0.300, 0.300])

    cfgm = cfgr['Y-E-915'] = {}
    cfgm['cite'] = r' \cite{saertexye915}'
    cfgm['mA_G'] = 0.915  # kg/m**2
    cfgm['mA_F'] = 0.909  # kg/m**2
    cfgm['mA_T'] = 0.006  # kg/m**2
    cfgm['mA_Fs'] = np.array([0.425, 0.0, 0.242, 0.242])

    cfgr2 = OrderedDict()
    cfgm = cfgr2['Krimmer'] = {}
    cfgm['rm_t'] = rm_t = 73.49E+6
    cfgm['rm_c'] = rm_c = 217.4E+6
    cfgm['rm_s'] = rm_s = 44.60E+6

    cfgm['EF_ref'] = EF_ref = 81.5E+09
    cfgm['nuF_ref'] = nuF_ref = 0.22
    cfgm['EM_ref'] = EM_ref = 3.08980E+09
    cfgm['nuM_ref'] = nuM_ref = 0.3681

    cfgm = cfgr2['WE91-1'] = {}
    cfgm['EM'] = 3.0E+09
    cfgm['RM'] = 76.0E+06
    cfgm['psi_M'] = 0.32

    EPS = 0.000000001
    cfg = OrderedDict()
    cfgm = cfg['EGL1600'] = {}
    cfgm['filename_db'] = 'EGL1600.db'
    cfgm['E1_target'] = 47E+09 * (0.512 / 0.6)  # 41.260E+09
    cfgm['E2_target'] = 9.3E+09  # 11.390E+09
    cfgm['G12_target'] = 4.4E+09  # 3.910E+09
    cfgm['mA_F'] = 1.6  # kg/m**2
    # cfgm['mA_tot'] = 2.4  # kg/m**2
    # cfgm['thickness'] = 1.243E-3  # thickness laminate
    cfgm['psi_M'] = cfgr2['WE91-1']['psi_M']  # matrix mass fraction
    cfgm['mA_T_lower'] = 0.000  # yarn kg/m**2
    cfgm['mA_T_upper'] = EPS  # 0.015  # yarn kg/m**2
    cfgm['rho_T'] = rho_T
    cfgm['rho_R'] = rho_R
    cfgm['fu'] = 0.95  # undulation correction factor
    cfgm['fkor'] = 1.00  # porosity correction factor
    cfgm['frac_0_lower'] = 0.90  # fraction of 0deg fibers
    cfgm['frac_0_upper'] = 1.0
    cfgm['frac_90_lower'] = 0.0  # fraction of 90deg fibers
    cfgm['frac_90_upper'] = 0.0  # notused, controlled by frac_0
    cfgm['rho_Gl'] = rho_Gl
    cfgm['EF'] = EF_ref
    cfgm['EM'] = EM_ref
    cfgm['nuF'] = nuF_ref
    cfgm['nuM'] = nuM_ref
    cfgm['RM'] = rm_t
    cfgm['lam_angles'] = np.array([0.0, 90.0])
    cfgm['opt_settings'] = {}
    cfgm['opt_settings']['PopSize'] = 100
    cfgm['opt_settings']['maxGen'] = 400
    cfgm['e11_t'] = 2.19E-02
    cfgm['e11_c'] = 1.60E-02
    cfgm['e22_t'] = 0.37E-02
    cfgm['g12'] = 1.50E-02
    cfgm['nuT'] = nuT
    cfgm['ET'] = ET
    cfgm['rho_S'] = rho_S
    cfgm['psi_SF'] = psi_SF
    cfgm['EF_ref'] = EF_ref
    cfgm['EM_ref'] = EM_ref
    cfgm['nuF_ref'] = nuF_ref
    cfgm['nuM_ref'] = nuM_ref
    cfgm['w_target'] = 0.999  # 0.999  # 0.999
    cfgm['fvf_lower'] = 0.5
    cfgm['fvf_upper'] = 0.6

    '''
    cfgm = cfg['XE600'] = {}
    cfgm['filename_db'] = 'XE600.db'
    cfgm['E1_target'] = 11.580E+09
    cfgm['E2_target'] = 11.580E+09
    cfgm['G12_target'] = 10.660E+09
    cfgm['mA_F'] = 0.603  # kg/m**2
    cfgm['mA_tot'] = 0.961  # kg/m**2
    cfgm['thickness'] = 0.514E-03  # thickness laminate
    cfgm['psi_M'] = 0.34  # matrix mass fraction
    cfgm['mA_T_lower'] = 0.002  # yarn kg/m**2
    cfgm['mA_T_upper'] = 0.015  # yarn kg/m**2
    cfgm['rho_T_lower'] = rho_T
    cfgm['rho_T_upper'] = rho_T + EPS
    cfgm['rho_R_lower'] = rho_R
    cfgm['rho_R_upper'] = rho_R + EPS
    cfgm['rho_Gl_lower'] = rho_Gl
    cfgm['rho_Gl_upper'] = rho_Gl + EPS
    cfgm['fu_lower'] = 0.97  # ondulation correction factor
    cfgm['fu_upper'] = 1.0
    cfgm['fkor_lower'] = 1.0  # porosity correction factor
    cfgm['fkor_upper'] = 1.01
    cfgm['frac_0_lower'] = 0.0  # fraction of 0deg fibers
    cfgm['frac_0_upper'] = EPS
    cfgm['frac_90_lower'] = 0.0  # fraction of 90deg fibers
    cfgm['frac_90_upper'] = EPS
    cfgm['EF_lower'] = 70.E+09
    cfgm['EF_upper'] = 85.E+09
    cfgm['EM_lower'] = 2.8E+09
    cfgm['EM_upper'] = 3.5E+09
    cfgm['nuF_lower'] = 0.16
    cfgm['nuF_upper'] = 0.25
    cfgm['nuM_lower'] = 0.32
    cfgm['nuM_upper'] = 0.40
    cfgm['RM'] = rm_t
    cfgm['lam_angles'] = np.array([0., 90., +45., -45.])
    cfgm['opt_settings'] = {}
    cfgm['opt_settings']['PopSize'] = 100
    cfgm['opt_settings']['maxGen'] = 400
    cfgm['e11_t'] = 1.07E-03
    cfgm['e11_c'] = 1.35E-03
    cfgm['e22_t'] = 1.35E-03
    cfgm['g12'] = 1.35E-03
    cfgm['nuT'] = nuT
    cfgm['ET'] = ET
    cfgm['rho_S'] = rho_S
    cfgm['psi_SF'] = psi_SF
    cfgm['EF_ref'] = EF_ref
    cfgm['EM_ref'] = EM_ref
    cfgm['nuF_ref'] = nuF_ref
    cfgm['nuM_ref'] = nuM_ref
    cfgm['w_target'] = 0.999
    '''

    cfgm = cfg['XE600'] = {}
    cfgm['filename_db'] = 'XE6004AX.db'
    cfgm['E1_target'] = 11.580E+09
    cfgm['E2_target'] = 11.580E+09
    cfgm['G12_target'] = 10.660E+09
    cfgm['mA_F'] = 0.603  # kg/m**2
    cfgm['mA_tot'] = 0.961  # kg/m**2
    cfgm['thickness'] = 0.514E-03  # thickness laminate
    cfgm['psi_M'] = 0.34  # matrix mass fraction
    cfgm['mA_T_lower'] = 0.002  # yarn kg/m**2
    cfgm['mA_T_upper'] = 0.006  # yarn kg/m**2
    cfgm['rho_T_lower'] = rho_T
    cfgm['rho_T_upper'] = rho_T + EPS
    cfgm['rho_R_lower'] = rho_R
    cfgm['rho_R_upper'] = rho_R + EPS
    cfgm['rho_Gl_lower'] = rho_Gl
    cfgm['rho_Gl_upper'] = rho_Gl + EPS
    cfgm['fu_lower'] = 0.85  # ondulation correction factor
    cfgm['fu_upper'] = 1.0
    cfgm['fkor_lower'] = 1.0  # porosity correction factor
    cfgm['fkor_upper'] = 1.01
    cfgm['frac_0_lower'] = 0.002  # fraction of 0deg fibers
    cfgm['frac_0_upper'] = 0.010  # EPS
    cfgm['frac_90_lower'] = 0.002  # fraction of 90deg fibers
    cfgm['frac_90_upper'] = 0.010  # EPS
    cfgm['EF_lower'] = 75.E+09
    cfgm['EF_upper'] = 87.E+09
    cfgm['EM_lower'] = 2.8E+09
    cfgm['EM_upper'] = 3.7E+09
    cfgm['nuF_lower'] = 0.19
    cfgm['nuF_upper'] = 0.23
    cfgm['nuM_lower'] = 0.35
    cfgm['nuM_upper'] = 0.38
    cfgm['RM'] = rm_t
    cfgm['lam_angles'] = np.array([0., 90., +45., -45.])
    cfgm['opt_settings'] = {}
    cfgm['opt_settings']['PopSize'] = 200
    cfgm['opt_settings']['maxGen'] = 400
    cfgm['e11_t'] = 1.07E-03
    cfgm['e11_c'] = 1.35E-03
    cfgm['e22_t'] = 1.35E-03
    cfgm['g12'] = 1.35E-03
    cfgm['nuT'] = nuT
    cfgm['ET'] = ET
    cfgm['rho_S'] = rho_S
    cfgm['psi_SF'] = psi_SF
    cfgm['EF_ref'] = EF_ref
    cfgm['EM_ref'] = EM_ref
    cfgm['nuF_ref'] = nuF_ref
    cfgm['nuM_ref'] = nuM_ref
    cfgm['w_target'] = 0.999

    '''
    cfgm = cfg['XE600S'] = {}
    cfgm['filename_db'] = 'XE600S.db'
    cfgm['E1_target'] = 12.750E+09
    cfgm['E2_target'] = 12.750E+09
    cfgm['G12_target'] = 10.660E+09
    cfgm['mA_F'] = 0.600  # kg/m**2
    cfgm['mA_tot'] = 0.896  # kg/m**2
    cfgm['thickness'] = 0.477E-03  # thickness laminate
    cfgm['psi_M'] = 0.32  # matrix mass fraction
    cfgm['mA_T_lower'] = 0.002  # yarn kg/m**2
    cfgm['mA_T_upper'] = 0.006  # yarn kg/m**2
    cfgm['rho_T_lower'] = rho_T
    cfgm['rho_T_upper'] = rho_T + EPS
    cfgm['rho_R_lower'] = rho_R
    cfgm['rho_R_upper'] = rho_R + EPS  # rho_R + EPS
    cfgm['rho_Gl_lower'] = rho_Gl
    cfgm['rho_Gl_upper'] = rho_Gl + EPS
    cfgm['fu_lower'] = 0.91  # 0.90  # ondulation correction factor
    cfgm['fu_upper'] = 1.0
    cfgm['fkor_lower'] = 1.001  # porosity correction factor
    cfgm['fkor_upper'] = 1.01
    cfgm['frac_0_lower'] = 0.0  # fraction of 0deg fibers
    cfgm['frac_0_upper'] = EPS
    cfgm['frac_90_lower'] = 0.0  # fraction of 90deg fibers
    cfgm['frac_90_upper'] = EPS
    cfgm['EF_lower'] = 79.E+09
    cfgm['EF_upper'] = 85.E+09  # 90.E+09
    cfgm['EM_lower'] = 2.6E+09
    cfgm['EM_upper'] = 3.5E+09
    cfgm['nuF_lower'] = 0.16
    cfgm['nuF_upper'] = 0.25
    cfgm['nuM_lower'] = 0.32
    cfgm['nuM_upper'] = 0.40
    cfgm['RM'] = rm_t
    cfgm['lam_angles'] = np.array([0., 90., +45.0, -45.])
    cfgm['opt_settings'] = {}
    cfgm['opt_settings']['PopSize'] = 100
    cfgm['opt_settings']['maxGen'] = 400
    cfgm['e11_t'] = 1.68E-03
    cfgm['e11_c'] = 1.45E-03
    cfgm['e22_t'] = 1.45E-03
    cfgm['g12'] = 1.35E-03
    cfgm['nuT'] = nuT
    cfgm['ET'] = ET
    cfgm['rho_S'] = rho_S
    cfgm['psi_SF'] = psi_SF
    cfgm['EF_ref'] = EF_ref
    cfgm['EM_ref'] = EM_ref
    cfgm['nuF_ref'] = nuF_ref
    cfgm['nuM_ref'] = nuM_ref
    cfgm['w_target'] = 0.999
    '''

    cfgm = cfg['XE600S'] = {}
    cfgm['filename_db'] = 'XE600S4AX.db'
    cfgm['E1_target'] = 12.750E+09
    cfgm['E2_target'] = 12.750E+09
    cfgm['G12_target'] = 10.660E+09
    cfgm['mA_F'] = 0.600  # kg/m**2
    cfgm['mA_tot'] = 0.896  # kg/m**2
    cfgm['thickness'] = 0.477E-03  # thickness laminate
    cfgm['psi_M'] = 0.32  # matrix mass fraction
    cfgm['mA_T_lower'] = 0.002  # yarn kg/m**2
    cfgm['mA_T_upper'] = 0.006  # yarn kg/m**2
    cfgm['rho_T_lower'] = rho_T
    cfgm['rho_T_upper'] = rho_T + EPS
    cfgm['rho_R_lower'] = rho_R
    cfgm['rho_R_upper'] = rho_R + EPS
    cfgm['rho_Gl_lower'] = rho_Gl
    cfgm['rho_Gl_upper'] = rho_Gl + EPS
    cfgm['fu_lower'] = 0.95  # 0.90  # ondulation correction factor
    cfgm['fu_upper'] = 1.0
    cfgm['fkor_lower'] = 0.90  # porosity correction factor
    cfgm['fkor_upper'] = 1.09
    cfgm['frac_0_lower'] = 0.002  # fraction of 0deg fibers
    cfgm['frac_0_upper'] = 0.010  # EPS
    cfgm['frac_90_lower'] = 0.002  # fraction of 90deg fibers
    cfgm['frac_90_upper'] = 0.010  # EPS
    cfgm['EF_lower'] = 75.0E+09
    cfgm['EF_upper'] = 87.E+09
    cfgm['EM_lower'] = 2.5E+09
    cfgm['EM_upper'] = 4.0E+09
    cfgm['nuF_lower'] = 0.19
    cfgm['nuF_upper'] = 0.23
    cfgm['nuM_lower'] = 0.35
    cfgm['nuM_upper'] = 0.38
    cfgm['RM'] = rm_t
    cfgm['lam_angles'] = np.array([0., 90., +45.0, -45.])
    cfgm['opt_settings'] = {}
    cfgm['opt_settings']['PopSize'] = 100
    cfgm['opt_settings']['maxGen'] = 400
    cfgm['e11_t'] = 1.68E-03
    cfgm['e11_c'] = 1.45E-03
    cfgm['e22_t'] = 1.45E-03
    cfgm['g12'] = 1.35E-03
    cfgm['nuT'] = nuT
    cfgm['ET'] = ET
    cfgm['rho_S'] = rho_S
    cfgm['psi_SF'] = psi_SF
    cfgm['EF_ref'] = EF_ref
    cfgm['EM_ref'] = EM_ref
    cfgm['nuF_ref'] = nuF_ref
    cfgm['nuM_ref'] = nuM_ref
    cfgm['w_target'] = 0.999

    cfgm = cfg['YE900'] = {}
    cfgm['filename_db'] = 'YE900.db'
    cfgm['E1_target'] = 20.260E+09
    cfgm['E2_target'] = 10.420E+09
    cfgm['G12_target'] = 7.352E+09
    cfgm['mA_F'] = 0.903  # kg/m**2
    cfgm['mA_tot'] = 1.46  # kg/m**2
    cfgm['thickness'] = 0.867E-03  # thickness laminate
    cfgm['psi_M'] = 0.39  # matrix mass fraction
    cfgm['mA_T_lower'] = 0.005  # yarn kg/m**2
    cfgm['mA_T_upper'] = 0.009  # yarn kg/m**2
    cfgm['rho_T_lower'] = rho_T
    cfgm['rho_T_upper'] = rho_T + EPS
    cfgm['rho_R_lower'] = rho_R
    cfgm['rho_R_upper'] = rho_R + EPS
    cfgm['rho_Gl_lower'] = rho_Gl
    cfgm['rho_Gl_upper'] = rho_Gl + EPS
    cfgm['fu_lower'] = 0.92  # 0.985  # ondulation correction factor
    cfgm['fu_upper'] = 1.0
    cfgm['fkor_lower'] = 1.0  # porosity correction factor
    cfgm['fkor_upper'] = 1.09
    cfgm['frac_0_lower'] = 0.38  # fraction of 0deg fibers
    cfgm['frac_0_upper'] = 0.48
    cfgm['frac_90_lower'] = 0.0  # fraction of 90deg fibers
    cfgm['frac_90_upper'] = EPS
    cfgm['EF_lower'] = 70.E+09
    cfgm['EF_upper'] = 82.E+09
    cfgm['EM_lower'] = 2.8E+09
    cfgm['EM_upper'] = 4.0E+09
    cfgm['nuF_lower'] = 0.19
    cfgm['nuF_upper'] = 0.23
    cfgm['nuM_lower'] = 0.35
    cfgm['nuM_upper'] = 0.38
    cfgm['RM'] = rm_t
    cfgm['lam_angles'] = np.array([0.0, 90.0, +45.0, -45.])
    cfgm['opt_settings'] = {}
    cfgm['opt_settings']['PopSize'] = 100
    cfgm['opt_settings']['maxGen'] = 400
    cfgm['e11_t'] = 2.33E-03
    cfgm['e11_c'] = 1.60E-03
    cfgm['e22_t'] = 1.22E-03
    cfgm['g12'] = 1.35E-03
    cfgm['nuT'] = nuT
    cfgm['ET'] = ET
    cfgm['rho_S'] = rho_S
    cfgm['psi_SF'] = psi_SF
    cfgm['EF_ref'] = EF_ref
    cfgm['EM_ref'] = EM_ref
    cfgm['nuF_ref'] = nuF_ref
    cfgm['nuM_ref'] = nuM_ref
    cfgm['w_target'] = 0.999

    '''
    cfg['YE900HRC'] = {}
    cfg['YE900HRC']['filename_db'] = 'YE900HRC.db'
    cfg['YE900HRC']['E1_target'] = 16.697E+09
    cfg['YE900HRC']['E2_target'] = 8.58752E+09
    cfg['YE900HRC']['G12_target'] = 6.605E+09
    cfg['YE900HRC']['mA_F'] = 0.903  # kg/m**2
    cfg['YE900HRC']['mA_tot'] = 1.46  # kg/m**2
    cfg['YE900HRC']['thickness'] = 1.052E-03  # thickness laminate
    cfg['YE900HRC']['psi_M'] = 0.44 * (1 - 0.03)  # matrix mass fraction
    cfg['YE900HRC']['mA_T_lower'] = 0.0  # sewing thread kg/m**2
    cfg['YE900HRC']['mA_T_upper'] = 0.01  # sewing thread kg/m**2
    cfg['YE900HRC']['rho_T_lower'] = 1200.
    cfg['YE900HRC']['rho_T_upper'] = 1600.
    cfg['YE900HRC']['fu_lower'] = 0.92  # ondulation correction factor
    cfg['YE900HRC']['fu_upper'] = 1.0
    cfg['YE900HRC']['fkor_lower'] = 0.92  # porosity correction factor
    cfg['YE900HRC']['fkor_upper'] = 1.0
    cfg['YE900HRC']['frac_0_lower'] = 0.2  # fraction of 0deg fibers
    cfg['YE900HRC']['frac_0_upper'] = 0.6
    cfg['YE900HRC']['frac_90_lower'] = 0.0  # fraction of 90deg fibers
    cfg['YE900HRC']['frac_90_upper'] = 0.0
    cfg['YE900HRC']['rho_Gl_lower'] = 2300.
    cfg['YE900HRC']['rho_Gl_upper'] = 2700.
    cfg['YE900HRC']['EF_lower'] = 65.E+09
    cfg['YE900HRC']['EF_upper'] = 80.E+09
    cfg['YE900HRC']['EM_lower'] = 2.0E+09
    cfg['YE900HRC']['EM_upper'] = 3.5E+09
    cfg['YE900HRC']['nuF_lower'] = 0.16
    cfg['YE900HRC']['nuF_upper'] = 0.25
    cfg['YE900HRC']['nuM_lower'] = 0.32
    cfg['YE900HRC']['nuM_upper'] = 0.45
    cfg['YE900HRC']['lam_angles'] = np.array([-45., 0.0, +45.])
    cfg['YE900HRC']['opt_settings'] = {}
    cfg['YE900HRC']['opt_settings']['PopSize'] = 100
    cfg['YE900HRC']['opt_settings']['maxGen'] = 400
    '''
    # eval_material()
    # eval_material_biax()

    m = 'EGL1600'
    optimize_material(cfg[m])
    post_optimization(cfg[m])
    #cfg['YE900HRC'] = vary_fvf(cfg['YE900'], thickness=1.052E-03)
    '''
    for m in cfg.iterkeys():
        # optimize_material(cfg[m])
        post_optimization(cfg[m])
    # write_result_file(cfg)
    '''
    write_result_file(cfg)
    write_property_file(cfg)
    write_layup_file(cfg, cfgr)
    #vary_fvf(cfg[m], cfg[m]['thickness'])

    write_constants_file(cfgc)
    write_stress_file(cfg, cfgr2)
