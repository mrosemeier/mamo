'''
Created on May 1, 2020

@author: Malo Rosemeier
'''

import os
import pkg_resources
import unittest
import tempfile
import shutil
from collections import OrderedDict
import matplotlib.pylab as plt
import pickle
import numpy as np
import pandas as pd
from mamo.models.sn import smax_basquin_goodman_weibull, SNFit,\
    sa_goodman, sa_basquin, sa_gerber, sa_loewenthal,\
    sa_swt, sa_tosa, sa_boerstra, smax_stuessi_boerstra_weibull, \
    smax_basquin_boerstra, smax_basquin_boerstra_weibull, R_ratio,\
    smax_sa, n_basquin_goodman_weibull, n_basquin_boerstra,\
    m_basquin_2p, m_basquin, dn_dsa_basquin, dn_dsa_stuessi,\
    n_basquin_boerstra_weibull, n_stuessi_boerstra_weibull, sa_stuessi, _b,\
    c_stuessi_orig, Na_stuessi_orig, sa_stuessi_orig_semilogx,\
    dsa_di_stuessi_orig_semilogx, d2sa_di2_stuessi_orig_semilogx, dsa_dn_stuessi
from mamo.models.normstren import ttg, Rt_norm_fiedler


PATH = pkg_resources.resource_filename('mamo', os.path.join('models', 'test'))


def normalize_sn_data(Td_tar=23.0, Tgd_tar=75.0):
    ''' Normalizes a corrected data set to a target design temperature 
    and a target glass transition temperature
    :param: td_tar: target design temperature in degree Celsius
    :param: tgd_tar: target design glass transition temperature in degree Celsius
    '''
    fname = 'tab01.dat'
    path_data = 'data'

    # Source: Rosemeier and Antoniou 2021, Tab. 1
    df = pd.read_csv(os.path.join(PATH, path_data, fname), escapechar='#')
    m = -371.3E6  # Source: Rosemeier and Antoniou 2021, Eq. 1
    dfe = df[df['runout'] == 0].reset_index()  # exclude runouts
    # normalize strengths to ttg_tar
    ttg_tar = ttg(Td_tar, Tgd_tar)
    sig_max_tar = Rt_norm_fiedler(
        dfe['ttg_cor'], ttg_tar, dfe['sig_max_cor'], m)
    dft = dfe.assign(ttg_tar=ttg_tar,
                     smax=sig_max_tar)

    return dft


def setup_fit(dff):

    # group data by stress ratio
    grp0_ids = dff[dff['R'] == 1.0].index.to_list()
    grp1_ids = dff[(dff['R'] <= 0.11) & (dff['R'] >= 0.09)].index.to_list()
    grp2_ids = dff[(dff['R'] <= -0.9) & (dff['R'] >= -1.1)].index.to_list()
    grp_entries = [grp0_ids, grp1_ids, grp2_ids]

    cfg = OrderedDict()
    cfg['m_start'] = 5.66
    cfg['Rt_start'] = 65.E+6
    cfg['Re_start'] = 13.E+6
    cfg['Na_start'] = 325
    cfg['N_fit_upper'] = 1E+8  # upper cycle bound for fitting
    cfg['p_tangent'] = 0.05  # SB quantile to tangent BBt
    return dff, grp_entries, cfg


def process_fit(dff, grp_entries, cfg):

    rst = OrderedDict()

    #######################################################################
    snf = SNFit(fit_type='stuessi-boerstra')
    rst[snf.fit_type] = snf
    #######################################################################
    snf.load_data(dff, grp_entries)
    snf.m_start = cfg['m_start']
    snf.Rt_start = cfg['Rt_start']
    snf.Re_start = cfg['Re_start']
    snf.Na_start = cfg['Na_start']
    snf.fit_data(cfg['N_fit_upper'])
    snf.tangent_basquin_boerstra(cfg['p_tangent'])

    #######################################################################
    snf = SNFit(fit_type='basquin-goodman')
    rst[snf.fit_type] = snf
    #######################################################################
    snf.load_data(dff, grp_entries)
    snf.m_start = cfg['m_start']
    snf.Rt_start = cfg['Rt_start']
    snf.fit_sn()
    snf.project_data(cfg['N_fit_upper'])
    snf.fit_cld()

    #######################################################################
    snf = SNFit(fit_type='basquin-boerstra')
    rst[snf.fit_type] = snf
    #######################################################################
    snf.load_data(dff, grp_entries)
    snf.m_start = cfg['m_start']
    snf.Rt_start = cfg['Rt_start']
    snf.Na_start = cfg['Na_start']
    snf.fit_data(cfg['N_fit_upper'])

    return rst


def normalize_fit_paper_data():
    # normalize data
    df = normalize_sn_data(Td_tar=23.0, Tgd_tar=86.1)
    # fit data
    dff, grp_entries, cfg = setup_fit(df)
    rst = process_fit(dff, grp_entries, cfg)
    return rst


def get_markov(path_data):
    # Markov matrices obtained via rainflow-counting of bending moment time
    # series obtained from aero-servo-elastic load simulations using
    # Modelica Library for Wind Turbines (MoWiT) for the DTU 10 MW RWT, DLC 1.1
    # Bending moment histories were transformed into strain histories at the
    # blade division points via Euler-Bernoulli beam theory
    # Further information about the DTU RWT:
    # http://www.innwind.eu/-/media/Sites/innwind/Publications/Deliverables/DeliverableD1-21ReferenceWindTurbinereport_INNWIND-EU.ashx
    # Further information about the used beam model:
    # https://doi.org/10.5281/zenodo.1494044

    rm_epoxy = 73.5E6  # epoxy strength (used to de-normalize stress exposure)
    v_ave_hub = 0.2 * 50.
    v = np.arange(3, 25 + 2, 2)

    z_pos = np.array([0, 5.37977, 11.1031, 17.0953, 23.2647, 29.5076,
                      35.7156, 41.7824, 47.6111, 53.1201, 58.247, 62.9501,
                      67.2076, 71.0159, 74.386, 77.3402, 79.9085, 82.1252,
                      84.0265, 85.6487, 86.366])

    design_life = 25
    d_time = design_life * 365.25 * 24 * 60 * 60  # seconds of 20 years
    time_sim = 600.  # sec
    d_time_factor = d_time / time_sim  # time series length

    def rayleigh_probability_density(v, v_ave_hub):
        '''
        Gasch, 2011, Windkraftanlagen, Eq. 4.13
        '''
        p = 0.5 * np.pi * v / v_ave_hub ** 2 * \
            np.exp(- 0.25 * np.pi * (v / v_ave_hub) ** 2)
        return p

    p = rayleigh_probability_density(v, v_ave_hub)
    delta = np.diff(v)[0]
    weight = p * delta
    np.sum(weight)

    fname = 'markov_dp0'
    with open(os.path.join(PATH, path_data, fname + '.pkl'), 'rb') as myrestoredata:
        m = pickle.load(myrestoredata)
    # [vwind, z, ('triax', 45)]

    material = 'triax'
    orientation = 45
    zidx = 4  # max chord
    markovs = []
    for vidx in range(len(v)):
        markov = m[v[vidx], zidx, material, orientation]
        markov[:, 2] = markov[:, 2] * weight[vidx]
        markovs.append(markov)

    markov_c = np.concatenate(markovs)

    sa = markov_c[:, 0] * rm_epoxy  # markov matrix contains stress exposure
    sm = markov_c[:, 1] * rm_epoxy  # markov matrix contains stress exposure
    n = markov_c[:, 2] * d_time_factor

    R = R_ratio(sm, sa)
    smax = smax_sa(sa, R)

    dfl = pd.DataFrame({'sa': sa,
                        'sm': sm,
                        'n': n,
                        'R': R,
                        'smax': smax,
                        })
    return dfl


class SNTestCase(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.path_data = 'data'
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp(dir=os.getcwd())

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_cld(self):

        self.assertAlmostEqual(
            sa_boerstra(sa0=1.0, sm=0.5, Rt=1.0, alp=1.0) /
            sa_goodman(sa0=1.0, sm=0.5, Rt=1.0),
            1.0, places=3)

        self.assertAlmostEqual(
            sa_boerstra(sa0=1.0, sm=0.5, Rt=1.0, alp=2.0) /
            sa_gerber(sa0=1.0, sm=0.5, Rt=1.0),
            1.0, places=3)

        Rt = 1.0
        m = 10
        exp_start = 0  # 10^0
        exp_end = 7  # 10^7
        npoint = 8
        ns = np.logspace(exp_start, exp_end, npoint)

        nsm = 100
        sa0 = sa_basquin(ns, m, Rt)
        sm = np.linspace(0., +1., nsm)

        sas_go = np.zeros((nsm, npoint))
        sas_ge = np.zeros((nsm, npoint))
        sas_lo = np.zeros((nsm, npoint))
        sas_swt = np.zeros((nsm, npoint))
        sas_tosa = np.zeros((nsm, npoint))
        sas_b = np.zeros((nsm, npoint))

        alp_start = 1.0
        m_alp = 30
        alp = sa_basquin(ns, m_alp, alp_start)

        for smi, sm_ in enumerate(sm):
            sas_go[smi, :] = sa_goodman(sa0, sm_, Rt)
            sas_ge[smi, :] = sa_gerber(sa0, sm_, Rt)
            sas_lo[smi, :] = sa_loewenthal(sa0, sm_, Rt)
            sas_swt[smi, :] = sa_swt(sa0, sm_)
            sas_tosa[smi, :] = sa_tosa(sa0, sm_, alp)
            sas_b[smi, :] = sa_boerstra(sa0, sm_, Rt, alp)

        fig, ax = plt.subplots()
        for i, _ in enumerate(ns):
            col = next(ax._get_lines.prop_cycler)['color']
            ax.plot(sm, sas_go[:, i], '--', color=col)
            ax.plot(sm, sas_ge[:, i], ':', color=col)
            #ax.plot(sm, sas_lo[:, i], '--', color=col)
            #ax.plot(sm, sas_swt[:, i], '--', color=col)
            #ax.plot(sm, sas_tosa[:, i], '--', color=col)
            ax.plot(sm, sas_b[:, i], '-', color=col)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.close(fig)

    def test_stuessi_orig_alt(self):

        exp_start = 0  # 10^0
        exp_end = 8  # 10^8
        ns = np.logspace(exp_start, exp_end, 1000)
        i = np.linspace(exp_start, exp_end, 1000)

        Na = 1E4
        Rt = 71.5E6
        Re = 22.3E6
        m = 10

        b = _b(m, Re, Rt)
        p = 1. / b

        c = c_stuessi_orig(Na, p)
        Na_calc = Na_stuessi_orig(c, p)

        self.assertAlmostEqual(Na_calc / Na, 1.0, places=3)

        sao = sa_stuessi_orig_semilogx(i, Rt, Re, c, p)
        dsao_dn = dsa_di_stuessi_orig_semilogx(i, Rt, Re, c, p)
        dsao_d2n = d2sa_di2_stuessi_orig_semilogx(i, Rt, Re, c, p)

        saa = sa_stuessi(ns, m, Rt, Re, Na, n0=0)
        dsaa_dn = dsa_dn_stuessi(ns, m, Rt, Re, Na, n0=0)

        sao_na = sa_stuessi_orig_semilogx(np.log10(Na), Rt, Re, c, p)
        saa_na = sa_stuessi(Na, m, Rt, Re, Na, n0=0)

        self.assertAlmostEqual(saa_na / sao_na, 1.0, places=3)

        fig, ax = plt.subplots()

        col = next(ax._get_lines.prop_cycler)['color']
        ax.plot(i, sao * 1E-6, linestyle='-',
                label=r'Stussi (orig)', color=col)
        ax.plot(i, dsao_dn * 1E-6, linestyle='--',
                label=r'Stussi 1st deriv. (orig)', color=col)
        ax.plot(i, dsao_d2n * 1E-6, linestyle='-.',
                label=r'Stussi 2nd deriv. (orig)', color=col)
        ax.plot(np.log10(Na),  sao_na * 1E-6, 'x', color=col)

        col = next(ax._get_lines.prop_cycler)['color']
        ax.plot(np.log10(ns), saa * 1E-6, linestyle='--',
                label=r'Stussi (alt)', color=col)
        ax.plot(np.log10(ns), dsaa_dn * 1E-6, linestyle=':',
                label=r'Stussi 1st deriv. (alt)', color=col)

        ax.plot(np.log10(Na),  saa_na * 1E-6, '+', color=col)

        ax.legend()
        plt.close(fig)

    def test_m_basquin(self):

        exp_start = 0  # 10^0
        exp_end = 8  # 10^7
        ns = np.logspace(exp_start, exp_end, 1000)

        Na = 1E4
        Rt = 71.5E6
        Re = 22.3E6
        m = 10
        smax_b = smax_basquin_boerstra(
            ns,
            R=-1,
            m=m,
            Rt=Rt,
            alp_c=[1., 0.],
            alp_fit='exp')
        smax_s = smax_stuessi_boerstra_weibull(
            ns,
            R=-1,
            m=m,
            Rt_fit=Rt,
            Re_fit=Re,
            Na=Na,
            alp_c=[1., 0., 0.],
            alp_fit='exp',
            p=0.50,
            lmbda=-5.6E6,
            delta=6.2E6,
            beta=3.9)
        smax_s_na = smax_stuessi_boerstra_weibull(
            Na,
            R=-1,
            m=m,
            Rt_fit=Rt,
            Re_fit=Re,
            Na=Na,
            alp_c=[1., 0., 0.],
            alp_fit='exp',
            p=0.50,
            lmbda=-5.6E6,
            delta=6.2E6,
            beta=3.9)

        # calculate partial derivate of a Basquin-like linear curve
        # through the inflection point of Stuessi's curve
        Ra = 0.5 * (Rt + Re)
        dn_dsa_b = dn_dsa_basquin(Na, Ra, Ra, m)
        dn_dsa_s = dn_dsa_stuessi(Ra, m, Rt, Re, Na)

        self.assertAlmostEqual(dn_dsa_b / dn_dsa_s, 1.0, places=3)

        # construct two points and test slope functions
        N1 = 100
        N2 = Na
        sa1 = smax_basquin_boerstra(
            N1,
            R=-1,
            m=m,
            Rt=Rt,
            alp_c=[1., 0.],
            alp_fit='exp')
        sa2 = smax_basquin_boerstra(
            N2,
            R=-1,
            m=m,
            Rt=Rt,
            alp_c=[1., 0.],
            alp_fit='exp')

        m_calc1 = m_basquin(Rt, sa1, N1)
        m_calc2 = m_basquin_2p(sa1, N1, sa2, N2)

        self.assertAlmostEqual(m_calc1 / m, 1.0, places=3)
        self.assertAlmostEqual(m_calc2 / m, 1.0, places=3)

        fig, ax = plt.subplots()

        col = next(ax._get_lines.prop_cycler)['color']
        ax.loglog(ns,
                  smax_b * 1E-6,
                  linestyle='-',
                  label=r'Basquin',
                  color=col)

        col = next(ax._get_lines.prop_cycler)['color']

        ax.loglog(ns, smax_s * 1E-6, linestyle='-',
                  label=r'Stuessi')

        col = next(ax._get_lines.prop_cycler)['color']
        ax.loglog(Na,  smax_s_na * 1E-6, '+', color=col)
        ax.loglog([ns[int(1E3 * 2 / 3)], ns[-1]],
                  np.r_[Re, Re] * 1E-6, ':', color=col)
        ax.loglog([ns[0], ns[int(1E3 * 1 / 3)]],
                  np.r_[Rt, Rt] * 1E-6, ':', color=col)

        plt.close(fig)

    def test_normalize(self):

        # normalize data
        df = normalize_sn_data(Td_tar=23.0, Tgd_tar=75.0)
        df.to_csv(os.path.join(PATH, self.test_dir, 'df_750.dat'))

        dsig_max_exp = -9.829785E6  # Rosemeier and Antoniou 2021, Fig2
        dsig_max = df['smax'] - df['sig_max_cor']

        self.assertAlmostEqual(dsig_max[0] / dsig_max_exp, 1.0, places=3)

    def test_fit(self):

        rst = normalize_fit_paper_data()

        for fit_type, snf in rst.items():
            # Rosemeier and Antoniou 2021, Fig4a
            if fit_type == 'basquin-goodman':
                m_fit_exp = 10.36771155004513
                self.assertAlmostEqual(
                    snf.sn_fit['m_fit'] / m_fit_exp, 1.0, places=3)

                Rt_fit_exp = 63312517.89557406
                self.assertAlmostEqual(
                    snf.sn_fit['Rt_fit'] / Rt_fit_exp, 1.0, places=3)

                alpha_exp = -13211256.353476021
                self.assertAlmostEqual(
                    snf.sn_fit['lmbda'] / alpha_exp, 1.0, places=3)

                beta_exp = 15352705.466147764
                self.assertAlmostEqual(
                    snf.sn_fit['delta'] / beta_exp, 1.0, places=3)

                gamma_exp = 2.503493845340379
                self.assertAlmostEqual(
                    snf.sn_fit['beta'] / gamma_exp, 1.0, places=3)

            # Rosemeier and Antoniou 2021, Fig4c
            elif fit_type == 'basquin-boerstra':
                m_fit_exp = 11.246422147405845
                self.assertAlmostEqual(
                    snf.sn_fit['m_fit'] / m_fit_exp, 1.0, places=3)

                Rt_fit_exp = 69027435.35715565
                self.assertAlmostEqual(
                    snf.sn_fit['Rt_fit'] / Rt_fit_exp, 1.0, places=3)

                alpha_exp = -3724357.2431028215
                self.assertAlmostEqual(
                    snf.sn_fit['lmbda'] / alpha_exp, 1.0, places=3)

                beta_exp = 4352356.440975934
                self.assertAlmostEqual(
                    snf.sn_fit['delta'] / beta_exp, 1.0, places=3)

                gamma_exp = 1.614829583823827
                self.assertAlmostEqual(
                    snf.sn_fit['beta'] / gamma_exp, 1.0, places=3)

            # Rosemeier and Antoniou 2021, Fig4e
            elif fit_type == 'stuessi-boerstra':
                m_fit_exp = 8.055004655716271
                self.assertAlmostEqual(
                    snf.sn_fit['m_fit'] / m_fit_exp, 1.0, places=3)

                Rt_fit_exp = 71444765.71436065
                self.assertAlmostEqual(
                    snf.sn_fit['Rt_fit'] / Rt_fit_exp, 1.0, places=3)

                Re_fit_exp = 22199143.85906476
                self.assertAlmostEqual(
                    snf.sn_fit['Re_fit'] / Re_fit_exp, 1.0, places=3)

                Na_exp = 162.3937737996702
                self.assertAlmostEqual(
                    snf.sn_fit['Na'] / Na_exp, 1.0, places=3)

                alpha_exp = -5584205.47332026
                self.assertAlmostEqual(
                    snf.sn_fit['lmbda'] / alpha_exp, 1.0, places=3)

                beta_exp = 6185211.1688987035
                self.assertAlmostEqual(
                    snf.sn_fit['delta'] / beta_exp, 1.0, places=3)

                gamma_exp = 3.885322972476399
                self.assertAlmostEqual(
                    snf.sn_fit['beta'] / gamma_exp, 1.0, places=3)

                # Rosemeier and Antoniou 2021, Fig5c
                n_test = 1E+5
                alp_exp = 0.43517844
                self.assertAlmostEqual(
                    snf.alp_c(np.log10(n_test)) / alp_exp, 1.0, places=3)

                # Rosemeier and Antoniou 2021, Fig5c, Eq.40
                a_exp = 0.7442852192598751
                d_exp = 0.12076292461453189
                self.assertAlmostEqual(snf.alpt_c[0] / a_exp, 1.0, places=3)
                self.assertAlmostEqual(snf.alpt_c[1] / d_exp, 1.0, places=3)

                # Rosemeier and Antoniou 2021, Fig5d
                Rt_p_exp = 68740726.64559755
                mt_exp = 9.332540710441792
                Nt_exp = 7275.341615163826
                self.assertAlmostEqual(
                    snf.sn_fit['Rt_p'] / Rt_p_exp, 1.0, places=3)
                self.assertAlmostEqual(
                    snf.sn_fit['mt'] / mt_exp, 1.0, places=3)
                self.assertAlmostEqual(
                    snf.sn_fit['Nt'] / Nt_exp, 1.0, places=3)

        # Rosemeier and Antoniou 2021, Fig5d
        fig, ax = plt.subplots()

        exp_start = 0
        exp_end = 8
        ns = np.logspace(exp_start, exp_end, 1000)
        p = 0.05  # failure probability

        for fit_type, snf in rst.items():
            ax.set_prop_cycle(None)  # reset cycler
            # ax.set_prop_cycle(custom_cycler)
            if not fit_type == 'basquin-boerstra':
                if fit_type == 'basquin-goodman':
                    linestyle = '-.'

                elif fit_type == 'stuessi-boerstra':
                    linestyle = '-'
                    linestyle_bb = ':'

                grps = snf.grps
                cyc_stress_max = snf.dff['smax']
                cyc_cycles = snf.dff['N']
                cyc_ratio_grp = snf.grps_ratio
                m_fit = snf.sn_fit['m_fit']
                Rt_fit = snf.sn_fit['Rt_fit']
                if fit_type == 'stuessi-boerstra':
                    Re_fit = snf.sn_fit['Re_fit']
                    Na_fit = snf.sn_fit['Na']
                    Rt_p = snf.sn_fit['Rt_p']
                    mt = snf.sn_fit['mt']
                    Nt = snf.sn_fit['Nt']
                if not fit_type == 'basquin-goodman':
                    alp_c = snf.alp_c
                    alp_fit = snf.alp_fit
                lmbda_fit = snf.sn_fit['lmbda']
                delta_fit = snf.sn_fit['delta']
                beta_fit = snf.sn_fit['beta']

                gidxs = range(len(grps))

                for gidx, grp in zip(gidxs, grps):
                    col = next(ax._get_lines.prop_cycler)['color']
                    if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves
                        if fit_type == 'basquin-goodman':
                            smax = smax_basquin_goodman_weibull(
                                ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit,
                                p=p, lmbda=lmbda_fit, delta=delta_fit, beta=beta_fit)
                        elif fit_type == 'stuessi-boerstra':
                            smax = smax_stuessi_boerstra_weibull(
                                ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit,
                                Re_fit=Re_fit, Na=Na_fit, alp_c=alp_c, alp_fit=alp_fit,
                                p=p, lmbda=lmbda_fit, delta=delta_fit, beta=beta_fit)
                        ax.semilogx(ns,
                                    smax * 1E-6,
                                    linestyle=linestyle,
                                    color=col)
                        # add basquin-boerstra tangent and tangent point
                        if fit_type == 'stuessi-boerstra':
                            smax_bbt = smax_basquin_boerstra(
                                ns,
                                R=cyc_ratio_grp[gidx],
                                m=mt,
                                Rt=Rt_p,
                                alp_c=snf.alpt_c,
                                alp_fit=snf.alpt_fit,)
                            st = smax_basquin_boerstra(
                                Nt,
                                R=cyc_ratio_grp[gidx],
                                m=mt,
                                Rt=Rt_p,
                                alp_c=snf.alpt_c,
                                alp_fit=snf.alpt_fit,)

                            ax.semilogx(ns,
                                        smax_bbt * 1E-6,
                                        linestyle=linestyle_bb,
                                        color=col)
                            ax.semilogx(Nt,
                                        st * 1E-6,
                                        linestyle='',
                                        marker='+',
                                        color=col)

                    # add measurement data points
                    for s, n in zip(cyc_stress_max[grp], cyc_cycles[grp]):
                        ax.semilogx(n, s * 1E-6, 'd', color=col, zorder=2.2)

                # add R=0.5
                R_ = 0.5
                col = next(ax._get_lines.prop_cycler)['color']
                if fit_type == 'basquin-goodman':
                    smax = smax_basquin_goodman_weibull(
                        ns, R=R_, m=m_fit, Rt_fit=Rt_fit,
                        p=p, lmbda=lmbda_fit, delta=delta_fit, beta=beta_fit)
                elif fit_type == 'stuessi-boerstra':
                    smax = smax_stuessi_boerstra_weibull(
                        ns, R=R_, m=m_fit, Rt_fit=Rt_fit,
                        Re_fit=Re_fit, Na=Na_fit, alp_c=alp_c, alp_fit=alp_fit,
                        p=p, lmbda=lmbda_fit, delta=delta_fit, beta=beta_fit)

                ax.semilogx(ns,
                            smax * 1E-6,
                            linestyle=linestyle,
                            color=col)

                # add basquin-boerstra tangent and tangent point
                if fit_type == 'stuessi-boerstra':
                    smax_bbt = smax_basquin_boerstra(
                        ns,
                        R=R_,
                        m=mt,
                        Rt=Rt_p,
                        alp_c=snf.alpt_c,
                        alp_fit=snf.alpt_fit)
                    st = smax_basquin_boerstra(
                        Nt,
                        R=R_,
                        m=mt,
                        Rt=Rt_p,
                        alp_c=snf.alpt_c,
                        alp_fit=snf.alpt_fit)

                    ax.semilogx(ns,
                                smax_bbt * 1E-6,
                                linestyle=linestyle_bb,
                                color=col)
                    ax.semilogx(Nt,
                                st * 1E-6,
                                linestyle='',
                                marker='+',
                                color=col)

        plt.close(fig)

    def test_cld_projection(self):

        rst = normalize_fit_paper_data()

        for fit_type, snf in rst.items():

            sm0 = Rt_fit = snf.sn_fit['Rt_fit']
            npoint = len(snf.alp)
            nsm = 1000
            sms = np.linspace(0., sm0, nsm)
            sas = np.zeros((nsm, npoint))

            for ni in range(npoint):
                if fit_type == 'basquin-boerstra' or fit_type == 'stuessi-boerstra':
                    sas[:, ni] = sa_boerstra(
                        snf.sa0[ni], sms, Rt_fit, snf.alp[ni])
                elif fit_type == 'basquin-goodman':
                    sas[:, ni] = sa_goodman(snf.sa0[ni], sms, Rt_fit)

            if fit_type == 'basquin-boerstra':
                sa_exp = 6895278.734190254
                self.assertAlmostEqual(
                    sas[500, 10] / sa_exp, 1.0, places=3)
            elif fit_type == 'stuessi-boerstra':
                sa_exp = 6363973.8597395765
                self.assertAlmostEqual(
                    sas[500, 10] / sa_exp, 1.0, places=3)
            elif fit_type == 'basquin-goodman':
                sa_exp = 10430952.79714319
                self.assertAlmostEqual(
                    sas[500, 10] / sa_exp, 1.0, places=3)

            fig, ax = plt.subplots()
            for ni in range(npoint):
                col = next(ax._get_lines.prop_cycler)['color']
                ax.plot(snf.sm_proj[:, ni] * 1E-6,
                        snf.sa_proj[:, ni] * 1E-6, 'd', color=col)
                ax.plot(sms * 1E-6, sas[:, ni] * 1E-6, color=col)
            plt.close(fig)

    def test_bbt_tangent(self):

        rst = normalize_fit_paper_data()

        smax_bbt_sb_exp = np.array(
            [1.00000004, 1.01699674, 1.01809091, 1.00831964, 0.99940094,
             0.99521156])

        Rs = [-1, -0.8, -0.5, 0.1, 0.5, 0.8]
        smax_sb = np.zeros_like(Rs)
        smax_bbt = np.zeros_like(Rs)
        for i, R in enumerate(Rs):
            smax_sb[i] = smax_stuessi_boerstra_weibull(
                p=0.05,
                n=rst['stuessi-boerstra'].sn_fit['Nt'],
                R=R,
                Rt_fit=rst['stuessi-boerstra'].sn_fit['Rt_fit'],
                m=rst['stuessi-boerstra'].sn_fit['m_fit'],
                Re_fit=rst['stuessi-boerstra'].sn_fit['Re_fit'],
                Na=rst['stuessi-boerstra'].sn_fit['Na'],
                lmbda=rst['stuessi-boerstra'].sn_fit['lmbda'],
                delta=rst['stuessi-boerstra'].sn_fit['delta'],
                beta=rst['stuessi-boerstra'].sn_fit['beta'],
                alp_c=rst['stuessi-boerstra'].alp_c,
                alp_fit=rst['stuessi-boerstra'].alp_fit
            )

            smax_bbt[i] = smax_basquin_boerstra(
                n=rst['stuessi-boerstra'].sn_fit['Nt'],
                R=R,
                m=rst['stuessi-boerstra'].sn_fit['mt'],
                Rt=rst['stuessi-boerstra'].sn_fit['Rt_p'],
                alp_c=rst['stuessi-boerstra'].alpt_c,
                alp_fit='exp')

            smax_bbt_sb = smax_bbt[i] / smax_sb[i]
            self.assertAlmostEqual(
                smax_bbt_sb / smax_bbt_sb_exp[i], 1.0, places=3)

    def test_smax_N(self):

        rst = normalize_fit_paper_data()

        p = 0.05
        smax_exp = 29986882.307854675
        n_exp = 1E4
        R = 0.1
        Rt_fit = rst['basquin-goodman'].sn_fit['Rt_fit']
        m_fit = rst['basquin-goodman'].sn_fit['m_fit']
        lmbda = rst['basquin-goodman'].sn_fit['lmbda']
        delta = rst['basquin-goodman'].sn_fit['delta']
        beta = rst['basquin-goodman'].sn_fit['beta']

        smax_ns_test = smax_basquin_goodman_weibull(
            n_exp, R, m_fit, Rt_fit, p, lmbda, delta, beta)
        n_smaxs_test = n_basquin_goodman_weibull(
            smax_exp, R, m_fit, Rt_fit, p, lmbda, delta, beta)

        self.assertAlmostEqual(smax_ns_test / smax_exp, 1.0, places=3)
        self.assertAlmostEqual(n_smaxs_test / n_exp, 1.0, places=3)

        # plot both curves against each other
        ns = np.logspace(0, 20, 1000)
        smaxs = np.linspace(0, 80E6, 1000)

        smax_ns = smax_basquin_goodman_weibull(
            ns, R, m_fit, Rt_fit, p, lmbda, delta, beta)
        n_smaxs = n_basquin_goodman_weibull(
            smaxs, R, m_fit, Rt_fit, p, lmbda, delta, beta)

        _, ax = plt.subplots()
        ax.semilogx(ns, smax_ns * 1E-6)
        ax.semilogx(n_smaxs, smaxs * 1E-6, '--')
        ax.set_xlim([1, 1E20])

        # calculate endurance limit for each stress ratio of the collective

        p = 0.05
        smax_exp = 30E6
        n_exp = 14843.252368456411
        R = 0.1
        Rt_fit = rst['basquin-boerstra'].sn_fit['Rt_fit']
        m_fit = rst['basquin-boerstra'].sn_fit['m_fit']
        lmbda = rst['basquin-boerstra'].sn_fit['lmbda']
        delta = rst['basquin-boerstra'].sn_fit['delta']
        beta = rst['basquin-boerstra'].sn_fit['beta']
        alp_c = rst['basquin-boerstra'].alp_c
        alp_fit = rst['basquin-boerstra'].alp_fit

        smax_ns_test = smax_basquin_boerstra_weibull(
            n_exp, R, m_fit, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta)
        n_smaxs_test = n_basquin_boerstra_weibull(
            smax_exp, R, m_fit, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta)

        self.assertAlmostEqual(smax_ns_test / smax_exp, 1.0, places=3)
        self.assertAlmostEqual(n_smaxs_test / n_exp, 1.0, places=3)

        # plot both curves against each other
        ns = np.logspace(0, 12, 1000)
        smaxs = np.linspace(0, 80E6, 1000)

        smax_ns = smax_basquin_boerstra_weibull(
            ns, R, m_fit, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta)
        n_smaxs = n_basquin_boerstra_weibull(
            smaxs, R, m_fit, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta)

        _, ax = plt.subplots()
        ax.semilogx(ns, smax_ns * 1E-6)
        ax.semilogx(n_smaxs, smaxs * 1E-6, '--')
        ax.set_xlim([1, 1E12])

        p = 0.05
        smax_exp = 29976146.92074468
        n_exp_aki = 7474.234839677993
        n_exp_exp = 7477.76333196836
        R = 0.1
        Rt_fit = rst['stuessi-boerstra'].sn_fit['Rt_fit']
        m_fit = rst['stuessi-boerstra'].sn_fit['m_fit']
        Re_fit = rst['stuessi-boerstra'].sn_fit['Re_fit']
        Na_fit = rst['stuessi-boerstra'].sn_fit['Na']
        lmbda = rst['stuessi-boerstra'].sn_fit['lmbda']
        delta = rst['stuessi-boerstra'].sn_fit['delta']
        beta = rst['stuessi-boerstra'].sn_fit['beta']

        # use_akima:
        alp_c = rst['stuessi-boerstra'].alp_c
        alp_fit = rst['stuessi-boerstra'].alp_fit

        smax_ns_test = smax_stuessi_boerstra_weibull(
            n_exp_aki, R, m_fit, Rt_fit, Re_fit, alp_c, alp_fit, Na_fit, p, lmbda, delta, beta)
        n_smaxs_test = n_stuessi_boerstra_weibull(
            smax_exp, R, m_fit, Rt_fit, Re_fit, alp_c, alp_fit, Na_fit, p, lmbda, delta, beta)

        self.assertAlmostEqual(smax_ns_test / smax_exp, 1.0, places=3)
        self.assertAlmostEqual(n_smaxs_test / n_exp_aki, 1.0, places=3)

        # use exp
        alp_fit = 'exp'
        alp_c = rst['stuessi-boerstra'].alpt_c

        smax_ns_test = smax_stuessi_boerstra_weibull(
            n_exp_exp, R, m_fit, Rt_fit, Re_fit, alp_c, alp_fit, Na_fit, p, lmbda, delta, beta)

        n_smaxs_test = n_stuessi_boerstra_weibull(
            smax_exp, R, m_fit, Rt_fit, Re_fit, alp_c, alp_fit, Na_fit, p, lmbda, delta, beta)

        self.assertAlmostEqual(smax_ns_test / smax_exp, 1.0, places=3)
        self.assertAlmostEqual(n_smaxs_test / n_exp_exp, 1.0, places=3)

        # plot both curves against each other
        ns = np.logspace(0, 10, 1000)
        smaxs = np.linspace(15E6, 80E6, 1000)

        #smax_ns = smax_basquin_boerstra_weibull(ns, R, m_fit, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta)
        smax_ns = smax_stuessi_boerstra_weibull(
            ns, R, m_fit, Rt_fit, Re_fit, alp_c, alp_fit, Na_fit, p, lmbda, delta, beta)
        n_smaxs = n_stuessi_boerstra_weibull(
            smaxs, R, m_fit, Rt_fit, Re_fit, alp_c, alp_fit, Na_fit, p, lmbda, delta, beta)

        _, ax = plt.subplots()
        ax.semilogx(ns, smax_ns * 1E-6)
        ax.semilogx(n_smaxs, smaxs * 1E-6, '-')
        ax.set_xlim([1, 1E20])

        smax_exp = 29999832.205295242
        n_exp = 2100.321516862882
        R = 0.1
        Rt_p = rst['stuessi-boerstra'].sn_fit['Rt_p']
        mt = rst['stuessi-boerstra'].sn_fit['mt']
        alp_fit = 'exp'
        alp_c = rst['stuessi-boerstra'].alpt_c

        smax_ns_test = smax_basquin_boerstra_weibull(
            n_exp, R, m_fit, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta)
        n_smaxs_test = n_basquin_boerstra_weibull(
            smax_exp, R, m_fit, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta)

        self.assertAlmostEqual(smax_ns_test / smax_exp, 1.0, places=3)
        self.assertAlmostEqual(n_smaxs_test / n_exp, 1.0, places=3)

        # plot both curves against each other
        ns = np.logspace(0, 40, 1000)
        smaxs = np.linspace(0, 80E6, 1000)

        smax_ns = smax_basquin_boerstra(ns, R, mt, Rt_p, alp_c, alp_fit)
        n_smaxs = n_basquin_boerstra(smaxs, R, mt, Rt_p, alp_c, alp_fit)

        _, ax = plt.subplots()
        ax.semilogx(ns, smax_ns * 1E-6)
        ax.semilogx(n_smaxs, smaxs * 1E-6, '--')
        ax.set_xlim([1, 1E12])

    def test_damage(self):

        # fit data
        rst = normalize_fit_paper_data()

        # get load collectives
        dfl = get_markov(self.path_data)

        # calculate permissible cycles for a load collective
        dfl['N_bg'] = n_basquin_goodman_weibull(
            smax=dfl['smax'].to_numpy(),
            R=dfl['R'].to_numpy(),
            m=rst['basquin-goodman'].sn_fit['m_fit'],
            Rt_fit=rst['basquin-goodman'].sn_fit['Rt_fit'],
            p=0.05,
            lmbda=rst['basquin-goodman'].sn_fit['lmbda'],
            delta=rst['basquin-goodman'].sn_fit['delta'],
            beta=rst['basquin-goodman'].sn_fit['beta'],
        )

        dfl['N_bbt'] = n_basquin_boerstra(
            smax=dfl['smax'].to_numpy(),
            R=dfl['R'].to_numpy(),
            m=rst['stuessi-boerstra'].sn_fit['mt'],
            Rt=rst['stuessi-boerstra'].sn_fit['Rt_p'],
            alp_c=rst['stuessi-boerstra'].alpt_c,
            alp_fit='exp'
        )

        print('Number of roots not found (N_bbt):', np.sum(
            np.isnan(dfl['N_bbt'].to_numpy())))

        dfl['N_sb'] = n_stuessi_boerstra_weibull(
            smax=dfl['smax'].to_numpy(),
            R=dfl['R'].to_numpy(),
            m=rst['stuessi-boerstra'].sn_fit['m_fit'],
            Rt_fit=rst['stuessi-boerstra'].sn_fit['Rt_fit'],
            Re_fit=rst['stuessi-boerstra'].sn_fit['Re_fit'],
            alp_c=rst['stuessi-boerstra'].alp_c,
            alp_fit='aki',
            Na=rst['stuessi-boerstra'].sn_fit['Na'],
            p=0.05,
            lmbda=rst['stuessi-boerstra'].sn_fit['lmbda'],
            delta=rst['stuessi-boerstra'].sn_fit['delta'],
            beta=rst['stuessi-boerstra'].sn_fit['beta'],
            n_end=1E14  # cycle number to determine endurance limit
        )

        print('Number of roots not found (N_sb):', np.sum(
            np.isnan(dfl['N_sb'].to_numpy())))
        # TODO: to avoid root finding issues, implement an akima1d interpolator,
        # which interpolates gaps that might occur

        dfl['D_sb'] = dfl['n'] / dfl['N_sb']
        dfl['D_bbt'] = dfl['n'] / dfl['N_bbt']
        dfl['D_bg'] = dfl['n'] / dfl['N_bg']

        D_bg_sum = np.sum(dfl['D_bg'].to_numpy())
        D_sb_sum = np.sum(dfl['D_sb'].to_numpy())
        D_bbt_sum = np.sum(dfl['D_bbt'].to_numpy())

        D_bg_sum_exp = 121.38093032386263
        D_sb_sum_exp = 44.344131595517936
        D_bbt_sum_exp = 159.1141012877871

        self.assertAlmostEqual(D_bg_sum / D_bg_sum_exp, 1.0, places=3)
        self.assertAlmostEqual(D_sb_sum / D_sb_sum_exp, 1.0, places=3)
        self.assertAlmostEqual(D_bbt_sum / D_bbt_sum_exp, 1.0, places=3)


if __name__ == '__main__':
    unittest.main()
