import os
import numpy as np
from mamo.models.sn import smax_stuessi_goodman_weibull, smax_stuessi_goodman,\
    x_weibull, _b, smax_basquin_goodman, smax_basquin_goodman_weibull, SNFit,\
    N_smax_stuessi_goodman_weibull, sa_smax, sm, sa_stuessi_weibull, CLDFit,\
    p_weibull, s_weibull, sa_goodman, sa_basquin, sa_gerber, sa_loewenthal,\
    sa_swt, sa_tosa, sa_boerstra, smax_stuessi_boerstra, N_smax_stuessi_boerstra,\
    explogx, N_stuessi, smax_stuessi_boerstra_weibull, poly1dlogx,\
    smax_basquin_boerstra, smax_basquin_boerstra_weibull

import matplotlib as mpl
from mamo.models.lib import readjson, writejson
import shutil
from collections import OrderedDict
import cPickle
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['text.latex.preamble'] = [
    r"\usepackage[utf8x]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{cmbright}",
    r"\usepackage{siunitx}",
    r"\usepackage{amsmath}",
]

width = 8.3E-2  # IOP 8.255E-2  # AIAA journal
mtoinch = 39.3700787402
hw = 6 / 8.
figsize = (width * mtoinch, hw * width * mtoinch)
fontsizeleg = 6  # 7  # 5  # 6
labelsize = 'x-small'  # 8
markersize = 3
pad = 0.03  # 0.035
handlelengthleg = 3
linewidth = 0.5
numpoints = 1
dpipng = 400
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.linewidth'] = linewidth
mpl.rcParams['patch.linewidth'] = linewidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['figure.figsize'] = figsize
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['axes.labelsize'] = labelsize
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['legend.numpoints'] = numpoints
mpl.rcParams['legend.fontsize'] = fontsizeleg
mpl.rcParams['legend.handlelength'] = handlelengthleg
mpl.rcParams['savefig.dpi'] = dpipng
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = pad
colors_10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

colors_20 = ['#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728',
             '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2',
             '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf',
             '#9edae5']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors_10)
import matplotlib.pylab as plt


def savefigdata(fig, filename):
    ''' Writes all plotted lines xy data of all axes to a text file
    '''
    for ai, ax in enumerate(fig.get_axes()):
        for li, line in enumerate(ax.get_lines()):
            header = 'ax_%02d_line_%02d' % (ai, li)
            if ai == 0 and li == 0:
                with open(filename, 'w') as write_file:
                    write_file.write('# ' + header + '\n')
            else:
                with open(filename, 'a') as write_file:
                    write_file.write('# ' + header + '\n')
            with open(filename, 'a') as write_file:
                xyd = line.get_xydata()
                np.savetxt(write_file, xyd)


def savefig(fig, folder, figname):
    fig.savefig(os.path.join(folder, figname + '.png'))
    fig.savefig(os.path.join(folder, figname + '.pdf'))
    fig.savefig(os.path.join(folder, figname + '.eps'))
    savefigdata(fig, os.path.join(folder, figname + '.dat'))


def color_map(number_of_lines, style=''):
    if style == 'fhg':
        colors = [(0.55294, 0.82745, 0.78039),
                  (0.95, 0.35, 0.70196),
                  (0.74509, 0.729411, 0.854901),
                  (0.98431, 0.50196, 0.44705),
                  (0.50196, 0.69411, 0.82745),
                  (0.99215, 0.70588, 0.38431),
                  (0.70196, 0.87058, 0.41176),
                  (0.98823, 0.80392, 0.89803),
                  (0.85098, 0.85098, 0.85098),
                  (0.73725, 0.50196, 0.74117),
                  (0.8, 0.92156, 0.772549),
                  (1, 0.92941, 0.43529),
                  (0.55294, 0.82745, 0.78039),
                  (0.95, 0.35, 0.70196),
                  (0.74509, 0.729411, 0.854901),
                  (0.98431, 0.50196, 0.44705),
                  (0.50196, 0.69411, 0.82745),
                  (0.99215, 0.70588, 0.38431),
                  (0.70196, 0.87058, 0.41176),
                  (0.98823, 0.80392, 0.89803),
                  (0.85098, 0.85098, 0.85098),
                  (0.73725, 0.50196, 0.74117),
                  (0.8, 0.92156, 0.772549),
                  (1, 0.92941, 0.43529),
                  (0.55294, 0.82745, 0.78039),
                  (0.95, 0.35, 0.70196),
                  (0.74509, 0.729411, 0.854901),
                  (0.98431, 0.50196, 0.44705),
                  (0.50196, 0.69411, 0.82745),
                  (0.99215, 0.70588, 0.38431),
                  (0.70196, 0.87058, 0.41176),
                  (0.98823, 0.80392, 0.89803),
                  (0.85098, 0.85098, 0.85098),
                  (0.73725, 0.50196, 0.74117),
                  (0.8, 0.92156, 0.772549),
                  (1, 0.92941, 0.43529),
                  ]
    else:
        cm = plt.get_cmap('jet')
        start = 0.2
        stop = 1.0
        colors = [cm(x) for x in np.linspace(start, stop, number_of_lines)]
    return colors


def _color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None


def _lab(pi, lab):
    if pi == 0:
        label = lab
    else:
        label = ''
    return label


def post_stuessi_goodman(fname, snf, ylim=[10, 80], legend=True, textbox=True, show_p5_p95=True, cldf=[]):

    writejson(snf.sn_fit, os.path.join(folder, fname + '_' + 'sn_fit_sg.json'))

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    cols = ['k', 'b', 'g', 'r', 'orange', 'm_fit']

    M_fit = 1
    lstyle = '-'

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_data['cyc_ratio_grp']
    m_fit = snf.sn_fit['m_fit']
    m_50 = snf.sn_fit['m_50']
    Rt_fit = snf.sn_fit['Rt_fit']
    Re_fit = snf.sn_fit['Re_fit']
    Na_fit = snf.sn_fit['Na']
    alp_smax_fit = snf.sn_fit['alpha']
    bet_smax_fit = snf.sn_fit['beta']
    gam_smax_fit = snf.sn_fit['gamma']
    Rt_50 = snf.sn_fit['Rt_50']
    Re_50 = snf.sn_fit['Re_50']
    Re_50_R = snf.sn_fit['Re_50_R']
    n0 = snf.n0

    #gidxs = [0, 1, 2, 3]
    gidxs = range(len(grps))

    if cldf:
        sfx = '_stuessi'
    else:
        sfx = '_weibull'

    #######################################################################
    figname = fname + '_' + 'sn_stuessi_goodman' + sfx
    #######################################################################
    fig, ax = plt.subplots()

    for gidx, grp, col in zip(gidxs, grps, cols):
        col = next(ax._get_lines.prop_cycler)['color']

        if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves

            show_fit_wo_weibull = False
            if show_fit_wo_weibull:
                smax_sg = smax_stuessi_goodman_weibull(ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt=Rt_fit, M=M_fit,
                                                       Re=Re_fit, Na=Na_fit)
                ax.semilogx(ns, smax_sg * 1E-6, linestyle=':', color=col,
                            label=r'Fit')

            if show_p5_p95:
                p = 0.05
                smax_05 = smax_stuessi_goodman_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                    Re_fit=Re_fit, Na=Na_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                ax.semilogx(ns, smax_05 * 1E-6, linestyle='-.', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            p = 0.50
            smax_50 = smax_stuessi_goodman_weibull(
                ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                Re_fit=Re_fit, Na=Na_fit,
                p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
            ax.semilogx(ns, smax_50 * 1E-6, linestyle='-', color=col,
                        label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))
            # label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100,
            # cyc_ratio_grp[gidx]))

            show_test_curve = False
            if show_test_curve:
                # p50% without weibull
                xw_50 = x_weibull(p=p, alpha=alp_smax_fit,
                                  beta=bet_smax_fit, gamma=gam_smax_fit)
                smax_50_wo = xw_50 + smax_stuessi_goodman(ns, R=cyc_ratio_grp[gidx], m=m_fit,
                                                          Rt=Rt_fit, M=M_fit, Re=Re_fit, Na=Na_fit, n0=n0)
                ax.semilogx(ns, smax_50_wo * 1E-6, linestyle='--', color='orange',
                            label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100, cyc_ratio_grp[gidx]))

            if show_p5_p95:
                p = 0.95
                smax_95 = smax_stuessi_goodman_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                    Re_fit=Re_fit, Na=Na_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                ax.semilogx(ns, smax_95 * 1E-6, linestyle='--', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            if cldf:
                if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves
                    gidx_s = gidx - 1
                    grps_s = cldf.sn_grp[gidx_s]['cyc_data']['grplist']
                    cyc_stress_max_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_stress_max']
                    cyc_cycles_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_cycles']
                    cyc_ratio_grp_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_ratio_grp']
                    m_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['m_fit']
                    Rt_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Rt_fit']
                    Re_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_fit']
                    Na_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Na']
                    alp_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['alpha']
                    bet_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['beta']
                    gam_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['gamma']
                    Rt_50_s = cldf.sn_grp[gidx_s]['sn_fit']['Rt_50']
                    Re_50_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_50']
                    Re_50_R_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_50_R']
                    n0_s = cldf.n0

                    p = 0.50
                    sa_50 = sa_stuessi_weibull(
                        ns, R=cyc_ratio_grp_s[1], m=m_fit_s, Rt_fit=Rt_fit_s, M=M_fit,
                        Re_fit=Re_fit_s, Na=Na_fit_s,
                        p=p, alpha=alp_smax_fit_s, beta=bet_smax_fit_s, gamma=gam_smax_fit_s, n0=n0_s)
                    ax.semilogx(ns, sa_50 * 1E-6, linestyle='--', color=col,
                                label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

    # for gidx, grp, col in zip(gidxs, grps, cols):
        for i, (s, n) in enumerate(zip(cyc_stress_max[grp], cyc_cycles[grp])):
            ax.semilogx(n, s * 1E-6, 'd',
                        color=col,  label=_lab(i, r'$R=%0.2f$' % (cyc_ratio_grp[gidx])))  # label=_lab(i, r'Exp.'))

    # place summary box
    textstr = '\n'.join((
        r'$m_\text{fit}=%.2f$' % (m_fit),
        r'$R^\text{t}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (Rt_fit * 1E-6),
        r'$R^\text{e}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (Re_fit * 1E-6),
        r'$N_\text{a}=%i$' % (Na_fit),
        r'$\alpha=\num{%.2E}$' % (alp_smax_fit),
        r'$\beta=\num{%.2E}$' % (bet_smax_fit),
        r'$\gamma=\num{%.2E}$' % (gam_smax_fit),
        r'$m^{R=-1}_{p=\SI{50}{\percent}}=%.2f$' % (m_50),
        r'$R^\text{t}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_50 * 1E-6),
        r'$R^{\text{e},R=-1}_{p=\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
            Re_50 * 1E-6)
    ))  # ,r'$R^{\text{e},R=0.1}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
    #     Re_50_R[gidx] * 1E-6)

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', edgecolor='black')

    if textbox:
        # place a text box in upper left in axes coords
        ax.text(0.97, 0.96, textstr, transform=ax.transAxes, fontsize=4,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

    ax.set_ylabel(smax_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(None, None)
    if legend:
        ax.legend(ncol=1, loc='lower left')
    ystep = 10
    yticks = np.arange(ylim[0], ylim[1] + ystep, ystep)
    ax.set_yticks(ticks=yticks, minor=False)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def post_stuessi(fname, cldf, xlim=[0, 7], ylim=[10, 80], legend=True, textbox=True, show_p5_p95=True):

    exp_start = xlim[0]  # 10^0
    exp_end = xlim[1]  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    cols = ['k', 'b', 'g', 'r', 'orange', 'm_fit']

    M_fit = 1
    lstyle = '-'

    #

    n0_s = cldf.n0

    #gidxs = [0, 1, 2, 3]
    gidxs = range(len(cldf.sn_grp))

    #######################################################################
    figname = fname + '_' + 'sn_stuessi'
    #######################################################################
    fig, ax = plt.subplots()

    for gidx_s in gidxs:
        col = next(ax._get_lines.prop_cycler)['color']

        #gidx_s = gidx - 1
        grps_s = cldf.sn_grp[gidx_s]['cyc_data']['grplist']
        cyc_stress_max_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_stress_max']
        cyc_cycles_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_cycles']
        cyc_ratio_grp_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_ratio_grp']
        m_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['m_fit']
        Rt_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Rt_fit']
        Re_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_fit']
        Na_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Na']
        alp_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['alpha']
        bet_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['beta']
        gam_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['gamma']
        Rt_50_s = cldf.sn_grp[gidx_s]['sn_fit']['Rt_50']
        Re_50_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_50']
        Re_50_R_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_50_R']

        if not cyc_ratio_grp_s[1] == 1:  # skip R=1 curves
            p = 0.50
            sa_50 = sa_stuessi_weibull(
                ns, R=cyc_ratio_grp_s[1], m=m_fit_s, Rt_fit=Rt_fit_s, M=M_fit,
                Re_fit=Re_fit_s, Na=Na_fit_s,
                p=p, alpha=alp_smax_fit_s, beta=bet_smax_fit_s, gamma=gam_smax_fit_s, n0=n0_s)
            ax.semilogx(ns, sa_50 * 1E-6, linestyle='--', color=col,
                        label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

        for gidx_s_ in grps_s:
            for i, (s, n) in enumerate(zip(cyc_stress_max_s[gidx_s_], cyc_cycles_s[gidx_s_])):
                # if not cyc_ratio_grp_s[0] == 1:
                ax.semilogx(n, s * 1E-6, 'd',
                            color=col,  label=_lab(i, r'$R=%0.2f$' % (cyc_ratio_grp_s[1])))  # label=_lab(i, r'Exp.'))

    ax.set_ylabel(smax_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(None, None)
    if legend:
        ax.legend(ncol=1, loc='lower left')
    ystep = 10
    yticks = np.arange(ylim[0], ylim[1] + ystep, ystep)
    ax.set_yticks(ticks=yticks, minor=False)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def post_stuessi_boerstra(fname, snf, xlim=[0, 7], ylim=[10, 80], legend=True, textbox=True, show_p5_p95=True, cldf=[]):

    writejson(snf.sn_fit, os.path.join(folder, fname + '_' + 'sn_fit_sb.json'))

    exp_start = xlim[0]  # 10^0
    exp_end = xlim[1]  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    cols = ['k', 'b', 'g', 'r', 'orange', 'm_fit']

    M_fit = 1
    lstyle = '-'

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_data['cyc_ratio_grp']
    m_fit = snf.sn_fit['m_fit']
    m_50 = snf.sn_fit['m_50']
    Rt_fit = snf.sn_fit['Rt_fit']
    Re_fit = snf.sn_fit['Re_fit']
    Na_fit = snf.sn_fit['Na']
    alp_c = snf.alp_c
    alp_fit = snf.alp_fit
    alp_smax_fit = snf.sn_fit['alpha']
    bet_smax_fit = snf.sn_fit['beta']
    gam_smax_fit = snf.sn_fit['gamma']
    Rt_50 = snf.sn_fit['Rt_50']
    Re_50 = snf.sn_fit['Re_50']
    Re_50_R = snf.sn_fit['Re_50_R']

    n0 = snf.n0

    #gidxs = [0, 1, 2, 3]
    gidxs = range(len(grps))

    if cldf:
        sfx = '_stuessi'
    else:
        sfx = '_weibull'

    #######################################################################
    figname = fname + '_' + 'sn_stuessi_boerstra' + sfx
    #######################################################################
    fig, ax = plt.subplots()

    for gidx, grp, col in zip(gidxs, grps, cols):
        col = next(ax._get_lines.prop_cycler)['color']

        if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves

            show_fit_wo_weibull = False
            if show_fit_wo_weibull:
                smax_sg = smax_stuessi_goodman_weibull(ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt=Rt_fit, M=M_fit,
                                                       Re=Re_fit, Na=Na_fit)
                ax.semilogx(ns, smax_sg * 1E-6, linestyle=':', color=col,
                            label=r'Fit')

            if show_p5_p95:
                p = 0.05
                smax_05 = smax_stuessi_boerstra_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit,
                    Re_fit=Re_fit, Na=Na_fit, alp_c=alp_c, alp_fit=alp_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                ax.semilogx(ns, smax_05 * 1E-6, linestyle='-.', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            p = 0.50
            smax_50 = smax_stuessi_boerstra_weibull(
                ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit,
                Re_fit=Re_fit, Na=Na_fit, alp_c=alp_c, alp_fit=alp_fit,
                p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
            ax.semilogx(ns, smax_50 * 1E-6, linestyle='-', color=col,
                        label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))
            # label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100,
            # cyc_ratio_grp[gidx]))

            if show_p5_p95:
                p = 0.95
                smax_95 = smax_stuessi_boerstra_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit,
                    Re_fit=Re_fit, Na=Na_fit, alp_c=alp_c, alp_fit=alp_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                ax.semilogx(ns, smax_95 * 1E-6, linestyle='--', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            if cldf:
                if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves
                    gidx_s = gidx - 1
                    grps_s = cldf.sn_grp[gidx_s]['cyc_data']['grplist']
                    cyc_stress_max_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_stress_max']
                    cyc_cycles_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_cycles']
                    cyc_ratio_grp_s = cldf.sn_grp[gidx_s]['cyc_data']['cyc_ratio_grp']
                    m_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['m_fit']
                    Rt_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Rt_fit']
                    Re_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_fit']
                    Na_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['Na']
                    alp_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['alpha']
                    bet_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['beta']
                    gam_smax_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['gamma']
                    Rt_50_s = cldf.sn_grp[gidx_s]['sn_fit']['Rt_50']
                    Re_50_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_50']
                    Re_50_R_s = cldf.sn_grp[gidx_s]['sn_fit']['Re_50_R']
                    n0_s = cldf.n0

                    p = 0.50
                    sa_50 = sa_stuessi_weibull(
                        ns, R=cyc_ratio_grp_s[1], m=m_fit_s, Rt_fit=Rt_fit_s, M=M_fit,
                        Re_fit=Re_fit_s, Na=Na_fit_s,
                        p=p, alpha=alp_smax_fit_s, beta=bet_smax_fit_s, gamma=gam_smax_fit_s, n0=n0_s)
                    ax.semilogx(ns, sa_50 * 1E-6, linestyle='--', color=col,
                                label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

    # for gidx, grp, col in zip(gidxs, grps, cols):
        for i, (s, n) in enumerate(zip(cyc_stress_max[grp], cyc_cycles[grp])):
            ax.semilogx(n, s * 1E-6, 'd',
                        color=col,  label=_lab(i, r'$R=%0.2f$' % (cyc_ratio_grp[gidx])))  # label=_lab(i, r'Exp.'))

    # place summary box
    textstr = '\n'.join((
        r'$m_\text{fit}=%.2f$' % (m_fit),
        r'$R^\text{t}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (Rt_fit * 1E-6),
        r'$R^\text{e}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (Re_fit * 1E-6),
        r'$N_\text{a}=%i$' % (Na_fit),
        r'$\alpha=\num{%.2E}$' % (alp_smax_fit),
        r'$\beta=\num{%.2E}$' % (bet_smax_fit),
        r'$\gamma=\num{%.2E}$' % (gam_smax_fit),
        r'$m^{R=-1}_{p=\SI{50}{\percent}}=%.2f$' % (m_50),
        r'$R^\text{t}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_50 * 1E-6),
        r'$R^{\text{e},R=-1}_{p=\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
            Re_50 * 1E-6)
    ))  # ,r'$R^{\text{e},R=0.1}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
    #     Re_50_R[gidx] * 1E-6)

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', edgecolor='black')

    if textbox:
        # place a text box in upper left in axes coords
        ax.text(0.97, 0.96, textstr, transform=ax.transAxes, fontsize=4,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

    ax.set_ylabel(smax_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(None, None)
    if legend:
        ax.legend(ncol=1, loc='lower left')
    ystep = 10
    yticks = np.arange(ylim[0], ylim[1] + ystep, ystep)
    ax.set_yticks(ticks=yticks, minor=False)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def post_basquin_goodmann(fname, snf, xlim=[0, 7], ylim=[10, 80], legend=True, textbox=True):

    writejson(snf.sn_fit, os.path.join(folder, fname + '_' + 'sn_fit_bg.json'))

    exp_start = xlim[0]  # 10^0
    exp_end = xlim[1]  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    cols = ['k', 'b', 'g', 'r', 'orange', 'm_fit']

    M_fit = 1
    lstyle = '-'

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_data['cyc_ratio_grp']
    m_fit = snf.sn_fit['m_fit']
    Rt_fit = snf.sn_fit['Rt_fit']
    alp_smax_fit = snf.sn_fit['alpha']
    bet_smax_fit = snf.sn_fit['beta']
    gam_smax_fit = snf.sn_fit['gamma']
    Rt_50 = snf.sn_fit['Rt_50']

    #gidxs = [0, 1, 2, 3]
    gidxs = range(len(grps))

    #######################################################################
    figname = fname + '_' + 'sn_basquin_goodman'
    #######################################################################
    fig, ax = plt.subplots()

    for gidx, grp, col in zip(gidxs, grps, cols):
        col = next(ax._get_lines.prop_cycler)['color']

        if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves

            show_fit_wo_weibull = False
            if show_fit_wo_weibull:
                smax_sg = smax_basquin_goodman(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt=Rt_fit, M=M_fit)
                ax.semilogx(ns, smax_sg * 1E-6, linestyle=lstyle, color=col,
                            label=r'Fit')

            show_p5_p95 = True
            if show_p5_p95:
                p = 0.05
                smax_05 = smax_basquin_goodman_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
                ax.semilogx(ns, smax_05 * 1E-6, linestyle='-.', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            p = 0.50
            smax_50 = smax_basquin_goodman_weibull(
                ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
            ax.semilogx(ns, smax_50 * 1E-6, linestyle='-', color=col,
                        label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))
            # label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100,
            # cyc_ratio_grp[gidx]))

            show_test_curve = False
            if show_test_curve:
                # p50% without weibull
                xw_50 = x_weibull(p=p, alpha=alp_smax_fit,
                                  beta=bet_smax_fit, gamma=gam_smax_fit)
                smax_50_wo = xw_50 + smax_basquin_goodman(ns, R=cyc_ratio_grp[gidx], m=m_fit,
                                                          Rt=Rt_fit, M=M_fit,)
                ax.semilogx(ns, smax_50_wo * 1E-6, linestyle='--', color='orange',
                            label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100, cyc_ratio_grp[gidx]))

                smax_50 = smax_basquin_goodman(ns, R=cyc_ratio_grp[gidx], m=m_fit,
                                               Rt=Rt_50, M=M_fit,)

                ax.semilogx(ns, smax_50 * 1E-6, linestyle='--', color='orange',
                            label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100, cyc_ratio_grp[gidx]))

            if show_p5_p95:
                p = 0.95
                smax_95 = smax_basquin_goodman_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
                ax.semilogx(ns, smax_95 * 1E-6, linestyle='--', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

    # for gidx, grp, col in zip(gidxs, grps, cols):
        for i, (s, n) in enumerate(zip(cyc_stress_max[grp], cyc_cycles[grp])):
            ax.semilogx(n, s * 1E-6, 'd',
                        color=col,  label=_lab(i, r'$R=%0.2f$' % (cyc_ratio_grp[gidx])))  # label=_lab(i, r'Exp.'))

    # place summary box
    textstr = '\n'.join((
        r'$m=%.2f$' % (m_fit),
        r'$R^\text{t}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_fit * 1E-6),
        r'$\alpha=\num{%.2E}$' % (alp_smax_fit),
        r'$\beta=\num{%.2E}$' % (bet_smax_fit),
        r'$\gamma=\num{%.2E}$' % (gam_smax_fit),
        r'$R^\text{t}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_50 * 1E-6)

    ))

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', edgecolor='black')

    if textbox:
        # place a text box in upper left in axes coords
        ax.text(0.97, 0.96, textstr, transform=ax.transAxes, fontsize=4,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

    ax.set_ylabel(smax_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(None, None)
    if legend:
        ax.legend(ncol=1, loc='lower left')
    ystep = 10
    yticks = np.arange(ylim[0], ylim[1] + ystep, ystep)
    ax.set_yticks(ticks=yticks, minor=False)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def post_basquin_boerstra(fname, snf, xlim=[0, 7], ylim=[10, 80], legend=True, textbox=True):

    writejson(snf.sn_fit, os.path.join(folder, fname + '_' + 'sn_fit_bb.json'))

    exp_start = xlim[0]  # 10^0
    exp_end = xlim[1]  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    cols = ['k', 'b', 'g', 'r', 'orange', 'm_fit']

    M_fit = 1
    lstyle = '-'

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_data['cyc_ratio_grp']
    m_fit = snf.sn_fit['m_fit']
    Rt_fit = snf.sn_fit['Rt_fit']
    alp_smax_fit = snf.sn_fit['alpha']
    bet_smax_fit = snf.sn_fit['beta']
    gam_smax_fit = snf.sn_fit['gamma']
    Rt_50 = snf.sn_fit['Rt_50']

    #gidxs = [0, 1, 2, 3]
    gidxs = range(len(grps))

    #######################################################################
    figname = fname + '_' + 'sn_basquin_boerstra'
    #######################################################################
    fig, ax = plt.subplots()

    for gidx, grp, col in zip(gidxs, grps, cols):
        col = next(ax._get_lines.prop_cycler)['color']

        if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves

            show_fit_wo_weibull = False
            if show_fit_wo_weibull:
                smax_sg = smax_basquin_boerstra(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt=Rt_fit, alp_c=snf.alp_c, alp_fit=snf.alp_fit)
                ax.semilogx(ns, smax_sg * 1E-6, linestyle=lstyle, color=col,
                            label=r'Fit')

            show_p5_p95 = True
            if show_p5_p95:
                p = 0.05
                smax_05 = smax_basquin_boerstra_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, alp_c=snf.alp_c, alp_fit=snf.alp_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
                ax.semilogx(ns, smax_05 * 1E-6, linestyle='-.', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            p = 0.50
            smax_50 = smax_basquin_boerstra_weibull(
                ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, alp_c=snf.alp_c, alp_fit=snf.alp_fit,
                p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
            ax.semilogx(ns, smax_50 * 1E-6, linestyle='-', color=col,
                        label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))
            # label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100,
            # cyc_ratio_grp[gidx]))

            show_test_curve = False
            if show_test_curve:
                # p50% without weibull
                xw_50 = x_weibull(p=p, alpha=alp_smax_fit,
                                  beta=bet_smax_fit, gamma=gam_smax_fit)
                smax_50_wo = xw_50 + smax_basquin_goodman(ns, R=cyc_ratio_grp[gidx], m=m_fit,
                                                          Rt=Rt_fit, M=M_fit,)
                ax.semilogx(ns, smax_50_wo * 1E-6, linestyle='--', color='orange',
                            label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100, cyc_ratio_grp[gidx]))

                smax_50 = smax_basquin_goodman(ns, R=cyc_ratio_grp[gidx], m=m_fit,
                                               Rt=Rt_50, M=M_fit,)

                ax.semilogx(ns, smax_50 * 1E-6, linestyle='--', color='orange',
                            label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100, cyc_ratio_grp[gidx]))

            if show_p5_p95:
                p = 0.95
                smax_95 = smax_basquin_boerstra_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, alp_c=snf.alp_c, alp_fit=snf.alp_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
                ax.semilogx(ns, smax_95 * 1E-6, linestyle='--', color=col,
                            label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

    # for gidx, grp, col in zip(gidxs, grps, cols):
        for i, (s, n) in enumerate(zip(cyc_stress_max[grp], cyc_cycles[grp])):
            ax.semilogx(n, s * 1E-6, 'd',
                        color=col,  label=_lab(i, r'$R=%0.2f$' % (cyc_ratio_grp[gidx])))  # label=_lab(i, r'Exp.'))

    # place summary box
    textstr = '\n'.join((
        r'$m=%.2f$' % (m_fit),
        r'$R^\text{t}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_fit * 1E-6),
        r'$\alpha=\num{%.2E}$' % (alp_smax_fit),
        r'$\beta=\num{%.2E}$' % (bet_smax_fit),
        r'$\gamma=\num{%.2E}$' % (gam_smax_fit),
        r'$R^\text{t}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_50 * 1E-6)

    ))

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', edgecolor='black')

    if textbox:
        # place a text box in upper left in axes coords
        ax.text(0.97, 0.96, textstr, transform=ax.transAxes, fontsize=4,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

    ax.set_ylabel(smax_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(None, None)
    if legend:
        ax.legend(ncol=1, loc='lower left')
    ystep = 10
    yticks = np.arange(ylim[0], ylim[1] + ystep, ystep)
    ax.set_yticks(ticks=yticks, minor=False)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def post_cld_fit(fname, cldf, grp_entries_list, snf=[]):

    ns = cldf.ns

    for i, _ in enumerate(grp_entries_list):
        writejson(cldf.sn_grp[i]['sn_fit'], os.path.join(folder, fname +
                                                         '_' + 'sn_fit' + '_%02d' % i))

    #######################################################################
    figname = fname + '_' + 'cld_fit'
    #######################################################################
    fig, ax = plt.subplots()

    for i, n in enumerate(ns):
        col = next(ax._get_lines.prop_cycler)['color']
        if snf:
            ax.plot(snf.sms[:, i] * 1E-6, snf.sas[:, i] * 1E-6,
                    '-', color=col)

        ax.plot(cldf.sms[:, i] * 1E-6, cldf.sas[:, i] * 1E-6,
                'o--', color=col, label=r'$N=\num{%0.0E}$' % (n))

    ax.set_ylabel(sa_label)
    ax.set_xlabel(sm_label)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)

    #######################################################################
    figname = fname + '_' + 'cld_alp_fit'
    #######################################################################
    fig, ax = plt.subplots()

    ax.semilogx(ns, cldf.alp, 'o', label=r'St{\"u}ssi fit')
    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    nsf = np.logspace(exp_start, exp_end, 1000)
    if cldf.alp_fit == 'exp':
        alpf = explogx(nsf, *cldf.alp_c)
    elif cldf.alp_fit == 'lin':
        alpf = poly1dlogx(nsf, *cldf.alp_c)
    elif cldf.alp_fit == 'aki':
        alpf = cldf.alp_c(np.log10(nsf))
    ax.semilogx(nsf, alpf, '-')
    # ,label=r'Fit $%0.2fx^{-%0.2f}+%0.2f$' % (cldf.alp_c[0], cldf.alp_c[1], cldf.alp_c[2]))

    ax.set_ylabel(alp_label)
    ax.set_xlabel(n_label)
    #ax.set_ylim(None, 1)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)

    plot_stuessi = False
    if plot_stuessi:
        for i, _ in enumerate(grp_entries_list):

            gidxs = [0, 1]

            cols = ['k', 'b', 'g', 'r', 'orange', 'm_fit']

            M_fit = 1
            lstyle = '-'

            grps = cldf.sn_grp[i]['cyc_data']['grplist']
            cyc_stress_max = cldf.sn_grp[i]['cyc_data']['cyc_stress_max']
            cyc_cycles = cldf.sn_grp[i]['cyc_data']['cyc_cycles']
            cyc_ratio_grp = cldf.sn_grp[i]['cyc_data']['cyc_ratio_grp']
            m_fit = cldf.sn_grp[i]['sn_fit']['m_fit']
            Rt_fit = cldf.sn_grp[i]['sn_fit']['Rt_fit']
            Re_fit = cldf.sn_grp[i]['sn_fit']['Re_fit']
            Na_fit = cldf.sn_grp[i]['sn_fit']['Na']
            alp_smax_fit = cldf.sn_grp[i]['sn_fit']['alpha']
            bet_smax_fit = cldf.sn_grp[i]['sn_fit']['beta']
            gam_smax_fit = cldf.sn_grp[i]['sn_fit']['gamma']
            Rt_50 = cldf.sn_grp[i]['sn_fit']['Rt_50']
            Re_50 = cldf.sn_grp[i]['sn_fit']['Re_50']
            Re_50_R = cldf.sn_grp[i]['sn_fit']['Re_50_R']
            n0 = cldf.n0

            ###################################################################
            figname = fname + '_' + 'cyclic_sn_stuessi_%02d' % i
            ###################################################################
            fig, ax = plt.subplots()

            for gidx, grp, col in zip(gidxs, grps, cols):

                if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves

                    show_fit_wo_weibull = False
                    if show_fit_wo_weibull:
                        smax_sg = smax_stuessi_goodman(ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt=Rt_fit, M=M_fit,
                                                       Re=Re_fit, Na=Na_fit)
                        ax.semilogx(ns, smax_sg * 1E-6, linestyle=':', color=col,
                                    label=r'Fit')

                    show_p5_p95 = True
                    if show_p5_p95:
                        p = 0.05
                        smax_05 = sa_stuessi_weibull(
                            ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                            Re_fit=Re_fit, Na=Na_fit,
                            p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                        ax.semilogx(ns, smax_05 * 1E-6, linestyle='-.', color=col,
                                    label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

                    p = 0.50
                    smax_50 = sa_stuessi_weibull(
                        ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                        Re_fit=Re_fit, Na=Na_fit,
                        p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                    ax.semilogx(ns, smax_50 * 1E-6, linestyle='-', color=col,
                                label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))
                    # label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100,
                    # cyc_ratio_grp[gidx]))

                    if show_p5_p95:
                        p = 0.95
                        smax_95 = sa_stuessi_weibull(
                            ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                            Re_fit=Re_fit, Na=Na_fit,
                            p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                        ax.semilogx(ns, smax_95 * 1E-6, linestyle='--', color=col,
                                    label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            for gidx, grp, col in zip(gidxs, grps, cols):
                for j, (s, n) in enumerate(zip(cyc_stress_max[grp], cyc_cycles[grp])):
                    ax.semilogx(n, s * 1E-6, 'd',
                                color=col,  label=_lab(j, r'$R=%0.2f$' % (cyc_ratio_grp[gidx])))  # label=_lab(i, r'Exp.'))

            # place summary box
            textstr = '\n'.join((
                r'$m=%.2f$' % (m_fit),
                r'$R^\text{t}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (
                    Rt_fit * 1E-6),
                r'$R^\text{e}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (
                    Re_fit * 1E-6),
                r'$N_\text{a}=%i$' % (Na_fit),
                r'$\alpha=\num{%.2E}$' % (alp_smax_fit),
                r'$\beta=\num{%.2E}$' % (bet_smax_fit),
                r'$\gamma=\num{%.2E}$' % (gam_smax_fit),
                r'$R^\text{t}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
                    Rt_50 * 1E-6),
                r'$R^{\text{e},R=-1}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
                    Re_50 * 1E-6),
                r'$R^{\text{e},R=0.1}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
                    Re_50_R[1] * 1E-6)
            ))

            # these are matplotlib.patch.Patch properties
            props = dict(facecolor='white', edgecolor='black')

            # place a text box in upper left in axes coords
            ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=4,
                    verticalalignment='top', bbox=props)

            ax.set_ylabel(smax_label)
            ax.set_xlabel(n_label)
            ax.set_ylim(10, 80)
            ax.set_xlim(None, None)
            #ax.legend(ncol=1, loc='lower left')
            ax.set_yticks(ticks=np.array(
                [10, 20, 30, 40, 50, 60, 70, 80]), minor=False)
            import matplotlib.ticker as ticker
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
            savefig(fig, folder=folder, figname=figname)
            plt.close(fig)


def post_weibull(fname, snf, legend=True, textbox=True):

    p_label = r'Probability $P$'
    Rt_label = r'Static strength $R^\text{t}$ in \si{\mega\pascal}'

    Rt_fit = snf.sn_fit['Rt_fit']
    alpha = snf.sn_fit['alpha']
    beta = snf.sn_fit['beta']
    gamma = snf.sn_fit['gamma']
    xs = snf.sn_fit['xs']

    x_min_ = x_weibull(0.001, alpha, beta, gamma)
    x_max_ = x_weibull(0.999, alpha, beta, gamma)
    xs_min_ = np.min(xs)
    xs_max_ = np.max(xs)

    xs_min = np.min([x_min_, xs_min_])
    xs_max = np.max([x_max_, xs_max_])

    mu, Rt_sig = s_weibull(alpha, beta, gamma)

    Rt_mean = mu + Rt_fit

    x = np.linspace(xs_min, xs_max, 1E3)
    p_fit = p_weibull(x, alpha, beta, gamma)

    #plt.plot((x + Rt_fit) * 1E-6, ps_fit)
    #plt.plot((xs + Rt_fit) * 1E-6, p_weibull(xs, alpha, beta, gamma), 'd')

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_data['cyc_ratio_grp']
    Rt_fit = snf.sn_fit['Rt_fit']
    alp_smax_fit = snf.sn_fit['alpha']
    bet_smax_fit = snf.sn_fit['beta']
    gam_smax_fit = snf.sn_fit['gamma']
    Rt_50 = snf.sn_fit['Rt_50']

    #gidxs = [0, 1, 2, 3]
    gidxs = range(len(grps))

    #######################################################################
    figname = fname + '_' + 'p_' + snf.fit_type
    #######################################################################
    fig, ax = plt.subplots()

    ax.plot((x + Rt_fit) * 1E-6, p_fit, 'k',
            label=r'$P\left(\alpha, \beta, \gamma\right)$')

    count = 0
    for gidx, grp in zip(gidxs, grps):
        col = next(ax._get_lines.prop_cycler)['color']

        for i, _ in enumerate(cyc_stress_max[grp]):
            xi = xs[count]
            pi = p_weibull(xi, alpha, beta, gamma)
            ax.plot((xi + Rt_fit) * 1E-6, pi, 'd',
                    color=col,  label=_lab(i, r'$R=%0.2f$' % (cyc_ratio_grp[gidx])))  # label=_lab(i, r'Exp.'))
            count += 1
    # place summary box
    textstr = '\n'.join((
        r'$\alpha=\num{%.2E}$' % (alp_smax_fit),
        r'$\beta=\num{%.2E}$' % (bet_smax_fit),
        r'$\gamma=\num{%.2E}$' % (gam_smax_fit),
        r'$R^\text{t}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_50 * 1E-6),
        r'$R^\text{t}_\text{mean}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_mean * 1E-6),
        r'$R^\text{t}_{\sigma}=\SI{%.2f}{\mega\pascal}$' % (
            Rt_sig * 1E-6)
    ))

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', edgecolor='black')

    if textbox:
        # place a text box in upper left in axes coords
        ax.text(0.03, 0.96, textstr, transform=ax.transAxes, fontsize=4,
                verticalalignment='top', horizontalalignment='left',
                bbox=props)

    ax.set_ylabel(p_label)
    ax.set_xlabel(Rt_label)
    #ax.set_ylim(ylim[0], ylim[1])
    #ax.set_xlim(None, None)
    if legend:
        ax.legend(ncol=1, loc='lower right')
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def post_cld(fname, snf, cfg, legend=True, textbox=True):
    #######################################################################
    figname = fname + '_' + 'cld_fit_' + snf.fit_type
    #######################################################################
    fig, ax = plt.subplots()

    nfilt = cfg['nfilt']

    for _, (i, n) in enumerate(zip(nfilt, snf.ns[nfilt])):
        col = next(ax._get_lines.prop_cycler)['color']
        ax.plot(snf.sm_proj[:, i] * 1E-6,
                snf.sa_proj[:, i] * 1E-6, 'd', color=col)

        ax.plot(snf.sms_b * 1E-6,
                snf.sas_b[:, i] * 1E-6,
                '-', color=col, label=r'$N=\num{%0.0E}$' % (n))

    ax.set_ylabel(sa_label)
    ax.set_xlabel(sm_label)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def post_alp(fname, snf, cfg, legend=True, textbox=True):
    #######################################################################
    figname = fname + '_' + 'cld_alp_fit_' + snf.fit_type
    #######################################################################
    fig, ax = plt.subplots()

    nfilt = cfg['nfilt']

    for _, (i, n) in enumerate(zip(nfilt, snf.ns[nfilt])):
        col = next(ax._get_lines.prop_cycler)['color']

        ax.semilogx(n, snf.alp[i],
                    'o', color=col, label=r'Boerstra fit')
    exp_start = 0  # 10^0
    exp_end = 8  # 10^7
    nsf = np.logspace(exp_start, exp_end, 1000)
    if snf.alp_fit == 'exp':
        alpf = explogx(nsf, *snf.alp_c)
    elif snf.alp_fit == 'lin':
        alpf = poly1dlogx(nsf, *snf.alp_c)
    elif snf.alp_fit == 'aki':
        alpf = snf.alp_c(np.log10(nsf))
    ax.semilogx(nsf, alpf, '--', color='k', label=r'Akima fit')
    # ,label=r'Fit $%0.2fx^{-%0.2f}+%0.2f$' % (cldf.alp_c[0], cldf.alp_c[1], cldf.alp_c[2]))

    ax.set_ylabel(alp_label)
    ax.set_xlabel(n_label)
    #ax.set_ylim(None, 1)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def process_fit(fname, data, grp_entries, grp_entries_list, cfg):

    exp_start = cfg['xlim'][0]
    exp_end = cfg['xlim'][1]
    npoint = cfg['npoint']
    ns = np.logspace(exp_start, exp_end, npoint)

    #######################################################################
    snf = SNFit(fit_type='basquin-boerstra')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = cfg['m_start']
    snf.Rt_start = cfg['Rt_start']
    snf.alp_c = cfg['alp_c']
    snf.alp_fit = cfg['alp_fit']
    snf.Na_start = cfg['Na_start']
    snf.fit_data(ns, cfg['include_weibull'], cfg['alp_idxs'])

    #######################################################################
    # Post Basquin-Boerstra
    #######################################################################
    ylim = cfg['ylim']
    xlim = cfg['xlim']
    legend = False
    textbox = True
    show_p5_p95 = False

    post_basquin_boerstra(fname, snf, xlim, ylim, legend, textbox)
    #post_cld_fit(fname, cldf, grp_entries_list, snf)
    post_weibull(fname, snf, legend=True, textbox=True)
    post_cld(fname, snf, cfg, legend=True, textbox=True)
    post_alp(fname, snf, cfg, legend=True, textbox=True)

    #######################################################################
    snf = SNFit(fit_type='basquin-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = cfg['m_start']
    snf.Rt_start = cfg['Rt_start']
    snf.fit_sn(cfg['include_weibull'])
    snf.project_data(ns)
    snf.fit_cld()

    #######################################################################
    # Post Basquin-Goodman
    #######################################################################

    post_basquin_goodmann(fname, snf, xlim, ylim, legend, textbox)
    post_weibull(fname, snf, legend=True, textbox=True)
    post_cld(fname, snf, cfg, legend=True, textbox=True)

    #######################################################################
    snf = SNFit(fit_type='stuessi-boerstra')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = cfg['m_start']
    snf.Rt_start = cfg['Rt_start']
    snf.Re_start = cfg['Re_start']
    snf.alp_c = cfg['alp_c']
    snf.alp_fit = cfg['alp_fit']  # cldf.alp_fit
    snf.Na_start = cfg['Na_start']
    snf.fit_data(ns, cfg['include_weibull'], cfg['alp_idxs'])
    #######################################################################
    # Post Stuessi-Boerstra
    #######################################################################

    post_stuessi_boerstra(fname, snf, xlim, ylim, legend,
                          textbox, show_p5_p95=True)
    post_weibull(fname, snf, legend=True, textbox=True)
    post_cld(fname, snf, cfg, legend=True, textbox=True)
    post_alp(fname, snf, cfg, legend=True, textbox=True)
    ####################


def fit_basquin_stuessi_spabond(npoint=8):

    fname = 'Spabond340_rev06'
    sfx = '.dat'
    path_data = 'data'
    path_file = os.path.join(path_data, fname + sfx)
    data = np.loadtxt(path_file, delimiter=',')
    data[:, 1] = data[:, 1] * 1E+6  # transform to SI
    data[:, 2] = data[:, 2] * 1E+6  # transform to SI

    if fname == 'Spabond340_rev00':
        grp0_ids = [0, 1, 2]  # R=1
        grp1_ids = [3, 4]  # R=0.6
        grp2_ids = [5, 6]  # R=0.1
        grp3_ids = [7, 8, 9]  # R=-1

        grp_entries = [grp0_ids, grp1_ids, grp2_ids, grp3_ids]

        grp_entries_list = [[grp0_ids, grp1_ids],
                            [grp0_ids, grp2_ids],
                            [grp0_ids, grp3_ids]]
    elif fname == 'Spabond340_rev04':
        grp0_ids = list(np.arange(0, 5))  # R=1
        grp11_ids = list(np.arange(5, 6))  # R=0.8
        grp12_ids = list(np.arange(6, 8))  # R=0.8
        grp21_ids = list(np.arange(8, 12))  # R=0.7...0.64
        grp22_ids = list(np.arange(12, 14))
        grp2f_ids = list(np.arange(7, 14))
        grp3_ids = list(np.arange(14, 16))  # R=0.1
        grp4_ids = list(np.arange(16, 19))  # R=-1

        grp_entries = [grp0_ids,
                       grp11_ids,
                       grp12_ids,
                       grp21_ids,
                       grp22_ids,
                       grp3_ids,
                       grp4_ids]  # ,grp1_ids,  grp3_ids

        grp_entries_list = [  # [grp0_ids, grp1_ids],
            [grp0_ids, grp2f_ids],
            [grp0_ids, grp3_ids],
            #[grp0_ids, grp4_ids]
        ]
    elif fname == 'Spabond340_rev05':
        grp0_ids = list(np.arange(0, 5))  # R=1
        grp11_ids = list(np.arange(10, 11))  # R=0.8
        grp12_ids = list(np.arange(11, 13))  # R=0.8
        grp21_ids = list(np.arange(13, 17))  # R=0.7...0.64
        grp22_ids = list(np.arange(17, 19))
        grp2f_ids = list(np.arange(12, 19))
        grp3_ids = list(np.arange(19, 21))  # R=0.1
        grp4_ids = list(np.arange(21, 24))  # R=-1

        grp_entries = [grp0_ids,
                       # grp11_ids,
                       grp12_ids,
                       grp21_ids,
                       grp22_ids,
                       grp3_ids,
                       grp4_ids]  # ,grp1_ids,  grp3_ids

        grp_entries_list = [  # [grp0_ids, grp1_ids],
            [grp0_ids, grp2f_ids],
            [grp0_ids, grp3_ids],
            #[grp0_ids, grp4_ids]
        ]

    elif fname == 'Spabond340_rev06':
        grp0_ids = list(np.arange(0, 22))  # R=1
        grp11_ids = list(np.arange(22, 23))  # R=0.8
        grp12_ids = list(np.arange(23, 25))  # R=0.8
        grp21_ids = list(np.arange(25, 29))  # R=0.7...0.64
        grp22_ids = list(np.arange(29, 31))
        grp2f_ids = list(np.arange(24, 31))
        grp3_ids = list(np.arange(31, 33))  # R=0.1
        grp4_ids = list(np.arange(33, 36))  # R=-1

        grp_entries = [grp0_ids,
                       grp11_ids,
                       grp12_ids,
                       grp21_ids,
                       grp22_ids,
                       grp3_ids,
                       grp4_ids]  # ,grp1_ids,  grp3_ids

        grp_entries_list = [  # [grp0_ids, grp1_ids],
            [grp0_ids, grp2f_ids],
            [grp0_ids, grp3_ids],
            #[grp0_ids, grp4_ids]
        ]

    cfg = OrderedDict()
    cfg['m_start'] = 6.0
    cfg['Rt_start'] = 40.E+6
    cfg['Re_start'] = 10.E+6
    cfg['Na_start'] = 1E3
    cfg['ylim'] = [10, 60]
    cfg['xlim'] = [0, 1]
    cfg['npoint'] = 5 * 9
    cfg['include_weibull'] = False
    cfg['alp_fit'] = 'exp'  # 'aki'
    cfg['alp_min'] = 0.475
    cfg['alp_c'] = [1.0, 0, 0]
    cfg['alp_idxs'] = []  # [0, 5, 6, 7, 8]  # [0, 1, 2]

    process_fit(fname, data, grp_entries, grp_entries_list, cfg)


def fit_basquin_stuessi_rim(npoint):

    fname = 'RIMR035c_rev03'
    sfx = '.dat'
    path_data = 'data'
    path_file = os.path.join(path_data, fname + sfx)
    data = np.loadtxt(path_file, delimiter=',')
    data[:, 1] = data[:, 1] * 1E+6  # transform to SI
    data[:, 2] = data[:, 2] * 1E+6  # transform to SI

    if fname == 'RIMR035c_rev01':
        grp0_ids = list(np.arange(0, 7))  # R=1 # exclude 50.39 and 59 MPa
        grp1_ids = list(np.arange(9, 19))  # R=0.1 incl runouts 20
        grp2_ids = list(np.arange(20, 27))  # R=-1 incl runouts 28

    elif fname == 'RIMR035c_rev02':
        grp0_ids = list(np.arange(0, 3))
        grp1_ids = list(np.arange(3, 13))
        grp2_ids = list(np.arange(13, 21))

    elif fname == 'RIMR035c_rev03':
        grp0_ids = list(np.arange(0, 4))
        grp1_ids = list(np.arange(5, 15))
        grp2_ids = list(np.arange(15, 23))

    grp_entries = [grp0_ids, grp1_ids, grp2_ids]
    grp_entries_list = [[grp0_ids, grp1_ids],
                        [grp0_ids, grp2_ids]]

    cfg = OrderedDict()
    cfg['m_start'] = 5.66
    cfg['Rt_start'] = 65.E+6
    cfg['Re_start'] = 13.E+6
    cfg['Na_start'] = 325
    cfg['ylim'] = [10, 80]
    cfg['xlim'] = [0, 8]
    cfg['npoint'] = 17
    cfg['include_weibull'] = False
    cfg['alp_fit'] = 'exp'
    cfg['alp_min'] = 0.40
    cfg['alp_c'] = [1.0, 0, 0]
    cfg['alp_idxs'] = [0,  6, 7, 8, 9, 10, 11,
                       12, 13, 14, 15, 16]  # [0, 1, 2]
    cfg['nfilt'] = [0, 6, 8, 10, 12]
    process_fit(fname, data, grp_entries, grp_entries_list, cfg)


def fit_basquin_stuessi_epon(npoint):

    fname = 'Epon826_rev00'
    sfx = '.dat'
    path_data = 'data'
    path_file = os.path.join(path_data, fname + sfx)
    data = np.loadtxt(path_file, delimiter=',')
    data[:, 1] = data[:, 1] * 1E+6  # transform to SI
    data[:, 2] = data[:, 2] * 1E+6  # transform to SI

    grp0_ids = list(np.arange(0, 5))  # R=1 factor 5 by copies
    grp1_ids = list(np.arange(5, 9))  # R=0.23...0.27 # exclude runaway
    grp11_ids = list(np.arange(5, 7))  # R=0.24
    grp12_ids = list(np.arange(7, 9))  # R=0.27 # exclude runaway
    grp01_ids = list(np.arange(10, 11))  # R=0.1
    grp2_ids = list(np.arange(11, 17))  # R=-0.1 # exclude runaway
    grp21_ids = list(np.arange(11, 13))  # R=-0.09...-0.1
    grp22_ids = list(np.arange(13, 17))  # R=-0.11...-0.13 # exclude runaway
    grp3_ids = list(np.arange(18, 22))  # R=-0.65 # exclude runout
    grp31_ids = list(np.arange(18, 20))  # R=-0.61...0.64
    grp32_ids = list(np.arange(20, 22))  # R=-0.61...0.64 # exclude runout
    grp4_ids = list(np.arange(23, 34))  # R=-0.9# exclude runout

    grp_entries = [
        grp0_ids,
        grp1_ids,  # grp11_ids, grp12_ids,
        grp2_ids,  # grp21_ids, grp22_ids,
        grp3_ids,  # grp31_ids, grp32_ids
    ]

    grp_entries_list = [[grp0_ids, grp1_ids],
                        [grp0_ids, grp2_ids],
                        [grp0_ids, grp3_ids]]

    cfg = OrderedDict()
    cfg['m_start'] = 8.5
    cfg['Rt_start'] = 85.E+6
    cfg['Re_start'] = 30.E+6
    cfg['Na_start'] = 39.
    cfg['ylim'] = [30, 90]
    cfg['npoint'] = npoint
    cfg['include_weibull'] = False
    cfg['alp_fit'] = 'lin'
    cfg['alp_idxs'] = [3, 4, 5, 6, 7]

    process_fit(fname, data, grp_entries, grp_entries_list, cfg)


def test_cld():

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
    sas_r = np.zeros((nsm, npoint))

    alp_start = 1.0
    m_alp = 30
    alp = sa_basquin(ns, m_alp, alp_start)

    for smi, sm_ in enumerate(sm):
        sas_go[smi, :] = sa_goodman(sa0, sm_, Rt)
        sas_ge[smi, :] = sa_gerber(sa0, sm_, Rt)
        sas_lo[smi, :] = sa_loewenthal(sa0, sm_, Rt)
        sas_swt[smi, :] = sa_swt(sa0, sm_, Rt)
        sas_tosa[smi, :] = sa_tosa(sa0, sm_, Rt, alp)
        sas_b[smi, :] = sa_boerstra(sa0, sm_, Rt, alp)

    #######################################################################
    figname = 'cld_test'
    #######################################################################
    fig, ax = plt.subplots()

    for i, n in enumerate(ns):
        col = next(ax._get_lines.prop_cycler)['color']
        #ax.plot(sm, sas_go[:, i], '-', color=col)
        #ax.plot(sm, sas_ge[:, i], '--', color=col)
        #ax.plot(sm, sas_lo[:, i], '--', color=col)
        #ax.plot(sm, sas_swt[:, i], '--', color=col)
        #ax.plot(sm, sas_tosa[:, i], '--', color=col)
        ax.plot(sm, sas_b[:, i], '-', color=col)
        #ax.plot(sm, sas_r[:, i], '-.', color=col)

    ax.set_ylabel(sa_label)
    ax.set_xlabel(sm_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def test_cld_fit(folder):

    with open('cld_fit_rim.pkl', 'rb') as myrestoredata:
        cldf = cPickle.load(myrestoredata)
    sm0 = cldf['sms'][0, 0]
    sa0 = cldf['sas'][-1, 0]
    cldf['sms'] = cldf['sms'] / sm0
    cldf['sas'] = cldf['sas'] / sa0

    nset = len(cldf['sms'])

    Rt = 1.0

    #m = 10

    ns = cldf['ns']
    npoint = len(ns)

    nsm = 1000
    #sa0 = sa_basquin(ns, m, Rt)
    sm = np.linspace(0., +1., nsm)

    sas_b = np.zeros((nsm, npoint))

    #alp_start = 1.0
    #m_alp = 30
    #alp = sa_basquin(ns, m_alp, alp_start)

    alp = np.zeros(npoint)
    for ni in range(npoint):
        xdata = sm_set = cldf['sms'][:, ni]
        ydata = sa_set = cldf['sas'][:, ni]
        sa0 = cldf['sas'][-1, ni]

        def func(x, alp):
            return sa_boerstra(sa0, x, Rt, alp)

        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(func, xdata, ydata)
        alp[ni] = popt[0]

        sas_b[:, ni] = sa_boerstra(sa0, sm, Rt, alp[ni])

    # fit over n
    def func2(x, a, b, c):
        return a * x**-b + c

    xdata = ns
    ydata = alp
    popt, pcov = curve_fit(func2, xdata, ydata)
    alp_opt = popt

    # fit over sa_R=-1
    import scipy.optimize as optimize

    def fit_N_stuessi(xdata, ydata,
                      m_start=11.,
                      Rt_start=1.,
                      Re_start=0.25,
                      Na_start=0.3,
                      n0=1.):
        initial_guess = [m_start, Na_start]

        def objective(params):
            m, Na = params  # multiple design variables
            crits = []

            def func(xi, m, Rt, Re, Na, n0):
                yi = Na + m * np.log10((Re - xi) / (xi - Rt + 1E-10))
                return yi

            for _, (xi, yact) in enumerate(zip(xdata, ydata)):
                '''
                Ni = N_stuessi(
                    xi, m=m, Rt=Rt_start,
                    Re=Re_start, Na=Na, n0=n0)
                '''

                yi = func(xi, m, Rt_start, Re_start, Na, n0)

                crit1 = (yi - yact)**2

                crits.append(crit1)
            '''
            import matplotlib.pyplot as plt
            #smax_ = np.linspace(smin_start, smax_start, 100)
            sa_ = np.linspace(Re_start, Rt_start, 100)
            fig, ax = plt.subplots()
            ax.plot(sa_, func(sa_, m, Rt_start, Re_start, Na, n0))
            #plt.ylim(-1., None)
            #plt.xlim(15E6, 15.1)
            '''
            critsum = np.sum(np.array(crits))  # objective function
            return critsum

        options = {}
        #options['xtol'] = 1E-1
        options['disp'] = False
        options['maxiter'] = 1E5
        options['maxfev'] = 1E5
        result = optimize.minimize(
            objective, initial_guess, method='Nelder-Mead', options=options)
        if result.success:
            fitted_params = result.x
        else:
            raise ValueError(result.message)

        m_fit,  Na_fit = fitted_params

        return m_fit, Rt_start, Re_start, Na_fit

    xdata = sas_b[0, :]
    alp_opt2 = fit_N_stuessi(xdata, ydata,
                             m_start=0.25,
                             Rt_start=1.,
                             Re_start=0.24,
                             Na_start=.5,
                             n0=1.)

    # def func3(x):
    #    return N_stuessi(sa=x, m=1.3, Rt=1.0, M=1, Re=0.25, Na=0.38, n0=1.0)
    #xdata = sas_b[0, :]
    #f3_opt, pcov = curve_fit(func3, xdata, ydata)
    #ppoly = np.polyfit(xdata, ydata, 3)
    #p = np.poly1d(ppoly)

    #######################################################################
    figname = 'cld_test'
    #######################################################################
    fig, ax = plt.subplots()

    for i, n in enumerate(ns):
        col = next(ax._get_lines.prop_cycler)['color']
        #ax.plot(sm, sas_go[:, i], '-', color=col)
        #ax.plot(sm, sas_ge[:, i], '--', color=col)
        #ax.plot(sm, sas_lo[:, i], '--', color=col)
        #ax.plot(sm, sas_swt[:, i], '--', color=col)
        #ax.plot(sm, sas_tosa[:, i], '--', color=col)
        ax.plot(sm, sas_b[:, i], '-', color=col)
        #ax.plot(sm, sas_r[:, i], '-.', color=col)

        ax.plot(cldf['sms'][:, i], cldf['sas'][:, i],
                'o--', color=col, label=r'$N=\num{%0.0E}$' % (n))

    ax.set_ylabel(sa_label)
    ax.set_xlabel(sm_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    npoint = 8
    nsf = np.logspace(exp_start, exp_end, 1000)

    #######################################################################
    figname = 'cld_test_alp'
    #######################################################################
    fig, ax = plt.subplots()

    ax.semilogx(ns, alp, 'o', label=r'Boerstra exponent $\alpha$')

    ax.semilogx(nsf, func2(nsf, *alp_opt), '-',
                label=r'Fit $%0.2fx^{-%0.2f}+%0.2f$' % (alp_opt[0], alp_opt[1], alp_opt[2]))

    ax.set_ylabel(alp_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(None, 1)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)

    #######################################################################
    figname = 'cld_test_alp_sa'
    #######################################################################
    fig, ax = plt.subplots()

    ax.plot(sas_b[0, :], alp, 'o', label=r'Boerstra exponent $\alpha$')

    ax.plot(sas_b[0, :], N_stuessi(alp, *alp_opt2), '-', label=r'Polyfit')

    # ax.semilogx(nsf, func2(nsf, *alp_opt), '-',
    # label=r'Fit $%0.2fx^{-%0.2f}+%0.2f$' % (alp_opt[0], alp_opt[1],
    # alp_opt[2]))

    ax.set_ylabel(alp_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(None, 1)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def test_stuessi_boerstra(folder):

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    npoint = 100
    ns = np.logspace(exp_start, exp_end, npoint)

    #nR = 10
    R = [-1.4, -1., 0.1]  # np.linspace(-1.,-1.1, 0.1, nR)
    nR = len(R)

    m = 4.6
    Rt = 60.E+6
    Re = 15.E+6
    Na = 1650
    #alp = 0.95
    M = 1.

    a_ = 0.63
    b_ = 0.18
    c_ = 0.37

    alp_c = [a_, b_, c_]

    alp_ = explogx(ns, a_, b_, c_)

    #alp_ = explogx(ns, a_, b_, c_)

    #smax_ = np.linspace(60E6, 15E6, 1000)
    #N_smax_stuessi_boerstra(smax_, R[0], m, Rt, Re, Na, alp=0.6, n0=1.0)

    smax = np.zeros((nR, npoint))
    smax_sb1 = np.zeros((nR, npoint))
    smax_sg = np.zeros((nR, npoint))
    for Ri, R_ in enumerate(R):
        smax[Ri, :] = smax_stuessi_boerstra(
            ns, R_, m, Rt, Re, Na, alp_c, n0=1.0)
        smax_sb1[Ri, :] = smax_stuessi_boerstra(
            ns, R_, m, Rt, Re, Na, [1., 0., 0.], n0=1.0)
        smax_sg[Ri, :] = smax_stuessi_goodman(
            ns, R_, m, Rt, M, Re, Na, n0=1.0)

    #######################################################################
    figname = 'sn_test'
    #######################################################################
    fig, ax = plt.subplots()

    for Ri, R_ in enumerate(R):
        col = next(ax._get_lines.prop_cycler)['color']
        ax.semilogx(ns, smax[Ri, :] * 1E-6, '-',
                    color=col, label=r'R=%0.2f' % R_)
        ax.semilogx(ns, smax_sb1[Ri, :] * 1E-6, '--',
                    color=col, label=r'R=%0.2f' % R_)
        ax.semilogx(ns, smax_sg[Ri, :] * 1E-6, '-.',
                    color=col, label=r'R=%0.2f' % R_)
    #ax.loglog(ns, smax * 1E-6)

    ax.set_ylabel(smax_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(10, 70)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


def test_basquin_boerstra(folder):

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    npoint = 100
    ns = np.logspace(exp_start, exp_end, npoint)

    #nR = 10
    R = [-1, 0.1, 1.0]  # np.linspace(-1.,-1.1, 0.1, nR)
    nR = len(R)

    m = 4.6
    Rt = 60.E+6
    Re = 15.E+6
    Na = 1650
    #alp = 0.95
    M = 1.

    a_ = 0.63
    b_ = 0.18
    c_ = 0.37

    alp_c = [a_, b_, c_]

    alp_ = explogx(ns, a_, b_, c_)

    #alp_ = explogx(ns, a_, b_, c_)

    #smax_ = np.linspace(60E6, 15E6, 1000)
    #N_smax_stuessi_boerstra(smax_, R[0], m, Rt, Re, Na, alp=0.6, n0=1.0)

    smax = np.zeros((nR, npoint))
    smax_bb = np.zeros((nR, npoint))
    for Ri, R_ in enumerate(R):
        smax[Ri, :] = smax_basquin_goodman(ns, R_, m, Rt, M=1.)
        smax_bb[Ri, :] = smax_basquin_boerstra(
            ns, R_, m, Rt, alp_c=alp_c)

    #######################################################################
    figname = 'sn_test'
    #######################################################################
    fig, ax = plt.subplots()

    for Ri, R_ in enumerate(R):
        col = next(ax._get_lines.prop_cycler)['color']
        ax.semilogx(ns, smax[Ri, :] * 1E-6, '-',
                    color=col, label=r'R=%0.2f' % R_)
        ax.semilogx(ns, smax_bb[Ri, :] * 1E-6, '--',
                    color=col, label=r'R=%0.2f' % R_)
    #ax.loglog(ns, smax * 1E-6)

    ax.set_ylabel(smax_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(10, 70)
    ax.legend()
    savefig(fig, folder=folder, figname=figname)
    plt.close(fig)


if __name__ == '__main__':

    smax_label = r'Max. stress $\sigma_\text{max}$ in \si{\mega \pascal}'
    n_label = r'Cycles to failure $N$'
    sa_label = r'Stress amplitude $\sigma_\text{a}$ in \si{\mega \pascal}'
    sm_label = r'Mean stress $\sigma_\text{m}$ in \si{\mega \pascal}'
    alp_label = r'Boerstra exponent $\alpha$'

    # test_cld_fit(folder='')
    # test_stuessi_boerstra(folder='')
    # test_basquin_boerstra(folder='')

    rev = '04'

    folder = '_result' + '_rev' + rev

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    npoint = 8
    # fit_basquin_stuessi_spabond(npoint)
    # fit_basquin_stuessi_epon(npoint)
    fit_basquin_stuessi_rim(npoint)
