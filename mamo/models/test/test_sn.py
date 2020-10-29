import os
import numpy as np
from mamo.models.sn import smax_stuessi_goodman_weibull, smax_stuessi_goodman,\
    x_weibull, _b, smax_basquin_goodman, smax_basquin_goodman_weibull, SNFit

import matplotlib as mpl
from mamo.models.lib import readjson, writejson
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


if __name__ == '__main__':

    n_label = r'Cycles to failure $N$'
    stress_max_label = r'Max. stress $\sigma_\text{max}$ in \si{\mega \pascal}'

    cyc_data = {}
    fname = 'RIMR035c_R01.dat'
    path_data = 'data'
    path_file = os.path.join(path_data, fname)
    data = np.loadtxt(path_file, delimiter=',')
    data[:, 1] = data[:, 1] * 1E+6  # transform to SI
    data[:, 2] = data[:, 2] * 1E+6  # transform to SI

    grp_entries = [9, 10]

    #######################################################################
    snf = SNFit(fit_type='stuessi-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 8.5
    snf.Rt_start = 67.E+6
    snf.Re_start = 9.E+6
    snf.Na_start = 68.
    snf.fit_data()
    writejson(snf.sn_fit, 'sn_fit_sg.json')
    sn_fit_sg = readjson('sn_fit_sg.json')

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    gidxs = [0, 1]

    cols = ['k', 'b', 'g', 'r', 'orange', 'm']

    M_fit = 1
    lstyle = '-'

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_ratio_grp
    m_fit = snf.sn_fit['m']
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

    #######################################################################
    figname = 'cyclic_sn_stuessi_goodman'
    #######################################################################
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

    show_Rmin1 = False
    if show_Rmin1:
        col = 'r'
        R = -1
        p = 0.05
        smax_05 = smax_stuessi_goodman_weibull(
            ns, R=R, m=m_fit, Rt_fit=Rt_fit, M=M_fit,
            Re_fit=Re_fit, Na=Na_fit,
            p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
        ax.semilogx(ns, smax_05 * 1E-6, linestyle='-.', color=col,
                    label=r'$P_{\SI{%i}{\percent}}$, $R=$%0.2f' % (p * 100, R))

        p = 0.50
        smax_50 = smax_stuessi_goodman_weibull(
            ns, R=R, m=m_fit, Rt_fit=Rt_fit, M=M_fit,
            Re_fit=Re_fit, Na=Na_fit,
            p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
        ax.semilogx(ns, smax_50 * 1E-6, linestyle='-', color=col,
                    label=r'$P_{\SI{%i}{\percent}}$, $R=$%0.2f' % (p * 100, R))

        # p50% without weibull
        b = _b(m=m_fit, Re=Re_fit, Rt=Rt_fit)
        b = _b(m=m_fit, Re=Re_50, Rt=Rt_50)
        xw_50 = x_weibull(p=p, alpha=alp_smax_fit,
                          beta=bet_smax_fit, gamma=gam_smax_fit)
        smax_50_wo = xw_50 + smax_stuessi_goodman(ns, R=R, m=m_fit,
                                                  Rt=Rt_fit, M=M_fit, Re=Re_fit, Na=Na_fit, n0=n0)
        ax.semilogx(ns, smax_50_wo * 1E-6, linestyle='--', color='orange',
                    label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100, cyc_ratio_grp[gidx]))

        p = 0.95
        smax_95 = smax_stuessi_goodman_weibull(
            ns, R=R, m=m_fit, Rt_fit=Rt_fit, M=M_fit,
            Re_fit=Re_fit, Na=Na_fit,
            p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
        ax.semilogx(ns, smax_95 * 1E-6, linestyle='--', color=col,
                    label=r'$P_{\SI{%i}{\percent}}$, $R=$%0.2f' % (p * 100, R))

    gidxs = [0, 1]
    for gidx, grp, col in zip(gidxs, grps, cols):
        for i, (s, n) in enumerate(zip(cyc_stress_max[grp], cyc_cycles[grp])):
            ax.semilogx(n, s * 1E-6, 'd',
                        color=col,  label=_lab(i, r'$R=%0.2f$' % (cyc_ratio_grp[gidx])))  # label=_lab(i, r'Exp.'))

    # place summary box
    textstr = '\n'.join((
        r'$m=%.2f$' % (m_fit),
        r'$R^\text{t}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (Rt_fit * 1E-6),
        r'$R^\text{e}_\text{fit}=\SI{%.2f}{\mega\pascal}$' % (Re_fit * 1E-6),
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
    ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)

    ax.set_ylabel(stress_max_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(10, 80)
    ax.set_xlim(None, None)
    ax.legend(ncol=1, loc='lower left')
    ax.set_yticks(ticks=np.array(
        [10, 20, 30, 40, 50, 60, 70, 80]), minor=False)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    savefig(fig, folder='', figname=figname)
    plt.close(fig)

    #######################################################################
    snf = SNFit(fit_type='basquin-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 10.0
    snf.Rt_start = 50.0E+6
    snf.fit_data()
    writejson(snf.sn_fit, 'sn_fit_bg.json')
    sn_fit_bg = readjson('sn_fit_bg.json')

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_ratio_grp
    m_fit = snf.sn_fit['m']
    Rt_fit = snf.sn_fit['Rt_fit']
    alp_smax_fit = snf.sn_fit['alpha']
    bet_smax_fit = snf.sn_fit['beta']
    gam_smax_fit = snf.sn_fit['gamma']
    Rt_50 = snf.sn_fit['Rt_50']

    #######################################################################
    figname = 'cyclic_sn_basquin_goodman'
    #######################################################################
    fig, ax = plt.subplots()

    for gidx, grp, col in zip(gidxs, grps, cols):

        if not cyc_ratio_grp[gidx] == 1:  # skip R=1 curves

            show_fit_wo_weibull = False
            if show_fit_wo_weibull:
                smax_sg = smax_basquin_goodman(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt=Rt_fit, M=M_fit)
                ax.loglog(ns, smax_sg * 1E-6, linestyle=lstyle, color=col,
                          label=r'Fit')

            show_p5_p95 = True
            if show_p5_p95:
                p = 0.05
                smax_05 = smax_basquin_goodman_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
                ax.loglog(ns, smax_05 * 1E-6, linestyle='-.', color=col,
                          label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

            p = 0.50
            smax_50 = smax_basquin_goodman_weibull(
                ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
            ax.loglog(ns, smax_50 * 1E-6, linestyle='-', color=col,
                      label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))
            # label=r'$P_{\SI{%i}{\percent}}$, $R=%0.2f$' % (p * 100,
            # cyc_ratio_grp[gidx]))

            if show_p5_p95:
                p = 0.95
                smax_95 = smax_basquin_goodman_weibull(
                    ns, R=cyc_ratio_grp[gidx], m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                    p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit)
                ax.loglog(ns, smax_95 * 1E-6, linestyle='--', color=col,
                          label=r'$P_{\SI{%i}{\percent}}$' % (p * 100))

    gidxs = [0, 1]
    for gidx, grp, col in zip(gidxs, grps, cols):
        for i, (s, n) in enumerate(zip(cyc_stress_max[grp], cyc_cycles[grp])):
            ax.loglog(n, s * 1E-6, 'd',
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

    # place a text box in upper left in axes coords
    ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)

    ax.set_ylabel(stress_max_label)
    ax.set_xlabel(n_label)
    ax.set_ylim(10, 80)
    ax.set_xlim(None, None)
    ax.legend(ncol=1, loc='lower left')
    ax.set_yticks(ticks=np.array(
        [10, 20, 30, 40, 50, 60, 70, 80]), minor=False)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    savefig(fig, folder='', figname=figname)
    plt.close(fig)
