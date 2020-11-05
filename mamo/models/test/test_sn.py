import os
import numpy as np
from mamo.models.sn import smax_stuessi_goodman_weibull, smax_stuessi_goodman,\
    x_weibull, _b, smax_basquin_goodman, smax_basquin_goodman_weibull, SNFit,\
    N_smax_stuessi_goodman_weibull, sa_smax, sm, sa_stuessi_weibull, CLDFit

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


def post_stuessi_goodman(fname, snf, ylim=[10, 80], legend=True, sntextbox=True, show_p5_p95=True, cldf=[]):

    writejson(snf.sn_fit, 'sn_fit_sg.json')
    sn_fit_sg = readjson('sn_fit_sg.json')

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    cols = ['k', 'b', 'g', 'r', 'orange', 'm']

    M_fit = 1
    lstyle = '-'

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_data['cyc_ratio_grp']
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

    #gidxs = [0, 1, 2, 3]
    gidxs = range(len(grps))

    #######################################################################
    figname = fname + '_' + 'sn_stuessi_goodman'
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
                    m_fit_s = cldf.sn_grp[gidx_s]['sn_fit']['m']
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
            Re_50 * 1E-6)
    ))  # ,r'$R^{\text{e},R=0.1}_{\SI{50}{\percent}}=\SI{%.2f}{\mega\pascal}$' % (
    #     Re_50_R[gidx] * 1E-6)

    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='white', edgecolor='black')

    if sntextbox:
        # place a text box in upper left in axes coords
        ax.text(0.97, 0.96, textstr, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

    ax.set_ylabel(stress_max_label)
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
    savefig(fig, folder='', figname=figname)
    plt.close(fig)


def post_basquin_godmann(fname, snf, ylim=[10, 80], legend=True, sntextbox=True):

    writejson(snf.sn_fit, 'sn_fit_bg.json')
    sn_fit_bg = readjson('sn_fit_bg.json')

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    ns = np.logspace(exp_start, exp_end, 1E3)

    cols = ['k', 'b', 'g', 'r', 'orange', 'm']

    M_fit = 1
    lstyle = '-'

    writejson(snf.sn_fit, 'sn_fit_bg.json')
    sn_fit_bg = readjson('sn_fit_bg.json')

    grps = snf.cyc_data['grplist']
    cyc_stress_max = snf.cyc_data['cyc_stress_max']
    cyc_cycles = snf.cyc_data['cyc_cycles']
    cyc_ratio_grp = snf.cyc_data['cyc_ratio_grp']
    m_fit = snf.sn_fit['m']
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

    if sntextbox:
        # place a text box in upper left in axes coords
        ax.text(0.97, 0.96, textstr, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

    ax.set_ylabel(stress_max_label)
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
    savefig(fig, folder='', figname=figname)
    plt.close(fig)


def post_cld_fit(fname, cldf, grp_entries_list, snf=[]):

    ns = cldf.ns

    for i, _ in enumerate(grp_entries_list):
        writejson(cldf.sn_grp[i]['sn_fit'], fname +
                  '_' + 'sn_fit' + '_%02d' % i)

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
    savefig(fig, folder='', figname=figname)
    plt.close(fig)

    for i, _ in enumerate(grp_entries_list):

        gidxs = [0, 1]

        cols = ['k', 'b', 'g', 'r', 'orange', 'm']

        M_fit = 1
        lstyle = '-'

        grps = cldf.sn_grp[i]['cyc_data']['grplist']
        cyc_stress_max = cldf.sn_grp[i]['cyc_data']['cyc_stress_max']
        cyc_cycles = cldf.sn_grp[i]['cyc_data']['cyc_cycles']
        cyc_ratio_grp = cldf.sn_grp[i]['cyc_data']['cyc_ratio_grp']
        m_fit = cldf.sn_grp[i]['sn_fit']['m']
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

        #######################################################################
        figname = fname + '_' + 'cyclic_sn_stuessi_%02d' % i
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
        ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', bbox=props)

        ax.set_ylabel(stress_max_label)
        ax.set_xlabel(n_label)
        ax.set_ylim(10, 80)
        ax.set_xlim(None, None)
        #ax.legend(ncol=1, loc='lower left')
        ax.set_yticks(ticks=np.array(
            [10, 20, 30, 40, 50, 60, 70, 80]), minor=False)
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        savefig(fig, folder='', figname=figname)
        plt.close(fig)


def fit_basquin_stuessi_spabond(npoint=8):

    fname = 'Spabond340_rev00'
    sfx = '.dat'
    path_data = 'data'
    path_file = os.path.join(path_data, fname + sfx)
    data = np.loadtxt(path_file, delimiter=',')
    data[:, 1] = data[:, 1] * 1E+6  # transform to SI
    data[:, 2] = data[:, 2] * 1E+6  # transform to SI

    grp0_ids = [0, 1, 2]  # R=1
    grp1_ids = [3, 4]  # R=0.6
    grp2_ids = [5, 6]  # R=0.1
    grp3_ids = [7, 8, 9]  # R=-1

    grp_entries = [grp0_ids, grp1_ids, grp2_ids, grp3_ids]

    #######################################################################
    snf = SNFit(fit_type='stuessi-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 6.0
    snf.Rt_start = 40.E+6
    snf.Re_start = 20.E+6
    snf.Na_start = 1E3
    snf.fit_data()

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    ns = np.logspace(exp_start, exp_end, npoint)

    snf.cld(ns)

    #######################################################################
    cldf = CLDFit()
    #######################################################################

    grp_entries_list = [[grp0_ids, grp1_ids],
                        [grp0_ids, grp2_ids],
                        [grp0_ids, grp3_ids]]

    cldf.m_start = 6.0
    cldf.Rt_start = 40.E+6
    cldf.Re_start = 15.E+6
    cldf.Na_start = 1E3
    cldf.fit_data_groups(ns, data, grp_entries_list)

    ylim = [10, 60]
    legend = False
    sntextbox = True
    show_p5_p95 = False
    post_stuessi_goodman(fname, snf, ylim, legend,
                         sntextbox, show_p5_p95, cldf)
    post_cld_fit(fname, cldf, grp_entries_list, snf)

    #######################################################################
    snf = SNFit(fit_type='basquin-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 4.8
    snf.Rt_start = 50.0E+6
    snf.fit_data()

    post_basquin_godmann(fname, snf, ylim, legend, sntextbox)


def fit_basquin_stuessi_rim(npoint):

    fname = 'RIMR035c_rev01'
    sfx = '.dat'
    path_data = 'data'
    path_file = os.path.join(path_data, fname + sfx)
    data = np.loadtxt(path_file, delimiter=',')
    data[:, 1] = data[:, 1] * 1E+6  # transform to SI
    data[:, 2] = data[:, 2] * 1E+6  # transform to SI

    grp0_ids = list(np.arange(0, 9))  # R=1
    grp1_ids = list(np.arange(9, 19))  # R=0.1 incl runouts 20
    grp2_ids = list(np.arange(20, 27))  # R=-1 incl runouts 28

    grp_entries = [grp0_ids, grp1_ids, grp2_ids]

    #######################################################################
    snf = SNFit(fit_type='stuessi-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 4.8
    snf.Rt_start = 67.E+6
    snf.Re_start = 9.E+6
    snf.Na_start = 68.
    snf.fit_data()

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    ns = np.logspace(exp_start, exp_end, npoint)

    snf.cld(ns)

    #######################################################################
    cldf = CLDFit()
    #######################################################################

    grp_entries_list = [[grp0_ids, grp1_ids],
                        [grp0_ids, grp2_ids]]

    cldf.m_start = 4.8
    cldf.Rt_start = 67.E+6
    cldf.Re_start = 9.E+6
    cldf.Na_start = 68.
    cldf.fit_data_groups(ns, data, grp_entries_list)

    ylim = [10, 70]
    legend = False
    sntextbox = True
    show_p5_p95 = False
    post_stuessi_goodman(fname, snf, ylim, legend,
                         sntextbox, show_p5_p95, cldf)
    post_cld_fit(fname, cldf, grp_entries_list, snf)

    #######################################################################
    snf = SNFit(fit_type='basquin-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 4.8
    snf.Rt_start = 67.0E+6
    snf.fit_data()

    post_basquin_godmann(fname, snf, ylim, legend, sntextbox)


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

    #######################################################################
    snf = SNFit(fit_type='stuessi-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 8.5
    snf.Rt_start = 85.E+6
    snf.Re_start = 30.E+6
    snf.Na_start = 39
    snf.fit_data()

    exp_start = 0  # 10^0
    exp_end = 7  # 10^7
    ns = np.logspace(exp_start, exp_end, npoint)

    snf.cld(ns)

    #######################################################################
    cldf = CLDFit()
    #######################################################################

    grp_entries_list = [[grp0_ids, grp1_ids],
                        [grp0_ids, grp2_ids],
                        [grp0_ids, grp3_ids]]

    cldf.m_start = 8.5
    cldf.Rt_start = 85.E+6
    cldf.Re_start = 30.E+6
    cldf.Na_start = 39
    cldf.fit_data_groups(ns, data, grp_entries_list)

    ylim = [30, 90]
    legend = False
    sntextbox = True
    show_p5_p95 = False
    post_stuessi_goodman(fname, snf, ylim, legend,
                         sntextbox, show_p5_p95, cldf)
    post_cld_fit(fname, cldf, grp_entries_list, snf)

    #######################################################################
    snf = SNFit(fit_type='basquin-goodman')
    #######################################################################
    snf.load_data(data, grp_entries)
    snf.m_start = 4.8
    snf.Rt_start = 80.0E+6
    snf.fit_data()

    post_basquin_godmann(fname, snf, ylim, legend, sntextbox)


if __name__ == '__main__':

    stress_max_label = r'Max. stress $\sigma_\text{max}$ in \si{\mega \pascal}'
    n_label = r'Cycles to failure $N$'
    sa_label = r'Stress amplitude $\sigma_\text{a}$ in \si{\mega \pascal}'
    sm_label = r'Mean stress $\sigma_\text{m}$ in \si{\mega \pascal}'

    npoint = 8
    # fit_basquin_stuessi_spabond(npoint)
    # fit_basquin_stuessi_rim(npoint)
    fit_basquin_stuessi_epon(npoint)
