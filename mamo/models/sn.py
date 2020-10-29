import numpy as np
import scipy.optimize as optimize
from scipy.special import gamma


def x_weibull(p, alpha, beta, gamma):
    ''' 
    Obains x value of a Weibull dsitribution for a given probability p
    (1) solve p = 1. - np.exp(-((x - alpha) / beta)**gamma) for x if beta>0
    and 
    (2) solve p = np.exp(-((x - alpha) / beta)**gamma) for x if beta<0
    Note: A negative beta flips the Weibull curve about x-axis
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: x: float value
    '''
    if bet < 0:
        x = alpha + beta * (-np.log(p))**(1. / gamma)
    else:
        x = alpha + beta * (-np.log(1. - p))**(1. / gamma)
    return x


def smax_basquin_goodman(n, R=-1, m=10, Rt=1.0, M=1):
    '''
    Max stress for given cycle number of an SN curve according to
    Basquin-Goodman
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :return: smax: float allowable maximum stress
    '''
    e_max = 2. / ((1. - R) * n**(1. / m) + M * (R + 1))
    return Rt * e_max


def Nallow_basquin_goodman(smax, R, m, Rt, M):
    '''
    Allowable cycles for given ma xstress of an SN curve according to
    Basquin-Goodman
    :param: smax: float max stress
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :return: Nallow: float allowable cycles
    '''
    e_max = smax / Rt
    return ((e_max * M * (R + 1.) - 2.) / (e_max * (R - 1.)))**m


def xrand_smax_basquin_goodman(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    return smax_i - smax_basquin_goodman(N_i, R_i, m_fit, Rt_fit, M_fit)


def smax_basquin_goodman_weibull(n, R, m, Rt_fit, M, p, alpha, beta, gamma):
    '''
    Max stress for given cycle number and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: smax: float allowable maximum stress
    '''
    return x_weibull(p, alpha, beta, gamma) +\
        smax_basquin_goodman(n, R, m, Rt_fit, M)


def smax_limit_basquin_goodman_weibull(p, Rt_fit, alpha, beta, gamma):
    '''
    Endurance limit as function of p and R
    :param: p: float probability
    :param: Rt_fit: float static strength (fit)
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: Rt: float static strength for arbitrary p
    '''
    xw_p = x_weibull(p, alpha, beta, gamma)
    Rt = xw_p + Rt_fit

    return Rt


def _b(m, Re, Rt):
    '''
    Transforms Basquin's m into Stuessi's b
    (1) Derive Basquin for s: dn/ds(s) = -(Na*m*(s/R_a)**(-m))/s
    (2) Derive Stuessi for s: dn/ds(s) = -(Na*b*(Re-Rt)*
                                    ((-Re+s)/(Rt-s))**(-b))/((Re-s)*(Rt-s))
    (3) Set R_a = 0.5*(Re+Rt)
    (4) Set (1)=(2) with (3) and solve for b
    '''
    return (m * (-Re + Rt)) / (2 * (Re + Rt))


def smax_stuessi_goodman(n, R=-1, m=10, Rt=1.0, M=1, Re=30, Na=1E3, n0=1.0):
    '''
    Max stress of an SN curve according to Stuessi-Goodman
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :param: Re: float endurance limit for R=-1
    :param: Na: float turning point
    :param: n0: intersection with smax axis
    :return: smax: float allowable maximum stress
    '''
    b = _b(m, Re, Rt)
    return (2. * Rt * (Re * ((n - n0) / Na)**(1. / b) + Rt)) / \
        (-Rt * (-1 + ((n - n0) / Na)**(1. / b) * (R - 1) - R *
                (M - 1.) - M) + Re * ((n - n0) / Na)**(1. / b) * (1. + R) * M)


def smax_limit_stuessi_goodman(R, Rt, Re, M=1):
    '''
    Endurance limit of function smax_stuessi_goodman for n->inf
    :param: R: float stress ratio
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: M: mean stress sensitivity
    :return: smax: float endurance limit for arbitrary R
    '''
    return (2. * Re * Rt) / (Re * M * (R + 1.) + Rt * (1. - R))


def smax_limit_stuessi_goodman_RdRt(R, Rt, Rd_Rt, M=1):
    '''
    Endurance limit of function smax_stuessi_goodman for n->inf
    :param: R: float stress ratio
    :param: Rt: float static strength
    :param: Rt_Rd: float ratio between static strength and endurance limit (R=-1)
    :param: M: mean stress sensitivity
    :return: smax: float endurance limit for arbitrary stress ratios R and Rd/Rt ratios
    '''
    return (2. * Rd_Rt * Rt) / (Rd_Rt * M * (R + 1.) + (1. - R))


def xrand_smax_stuessi_goodman(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit, Re_fit, Na_fit, n0=0.):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    return smax_i - smax_stuessi_goodman(N_i, R_i, m_fit, Rt_fit, M_fit, Re_fit, Na_fit, n0)


def smax_stuessi_goodman_weibull(n, R, m, Rt_fit, M, Re_fit, Na, p, alpha, beta, gamma, n0):
    '''
    Max stress for given cycle number and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: Re_fit: float endurance limit for R=-1 (fit)
    :param: Na: float turning point
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: smax: float allowable maximum stress
    '''
    return x_weibull(p, alpha, beta, gamma) +\
        smax_stuessi_goodman(n, R, m, Rt_fit, M, Re_fit, Na, n0)


def smax_limit_stuessi_goodman_weibull(p, R, Rt_fit, M, Re_fit, alpha, beta, gamma):
    '''
    Endurance limit as function of p and R
    :param: p: float probability
    :param: R: float stress ratio
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: Re_fit: float endurance limit for R=-1 (fit)
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: Rt: float static strength for arbitrary p
    :return: Re: float endurance limit for arbitrary R and p
    '''
    xw_p = x_weibull(p, alpha, beta, gamma)
    Re = xw_p + smax_limit_stuessi_goodman(R, Rt_fit, Re_fit, M)
    Rt = xw_p + Rt_fit

    return Rt, Re


def pwm_weibull(xs):
    '''
    Fits a three parameter Weibull distribution using PWM
    Source: Taosa Caiza and Ummenhofer 2011
    :param: xs: array of unsorted random variables
    :return: alp: float Weibull parameter
    :return: bet: float Weibull parameter
    :return: gam: float Weibull parameter
    '''
    n = len(xs)
    # sort xs ascending
    xs_sort = np.sort(xs)
    # eq. 19
    m0 = 1. / n * np.sum(xs_sort)
    # eq. 20
    sum1 = 0.
    for i in range(n):
        sum1 += (n - i + 1) * xs_sort[i]
    m1 = 1. / (n * (n - 1.)) * sum1
    # eq. 21
    sum2 = 0.
    for i in range(n - 1):
        sum2 += (n - i + 1) * (n - i) * xs_sort[i]
    m2 = 1. / (n * (n - 1.) * (n - 2.)) * sum2

    def objective(params):
        gam_ = params[0]
        # eq. 16
        crit = (3. * m2 - m0) / (2. * m1 - m0) - \
            (3.**(-1. / gam_) - 1.) / (2.**(-1. / gam_) - 1.)
        return crit
    gam_start = 0.5
    initial_guess = [gam_start]
    result = optimize.minimize(objective, initial_guess, method='SLSQP')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    gam = fitted_params[0]

    gamma_gam = gamma(1. + 1. / gam)
    # eq. 17
    bet = (2. * m1 - m0) / ((2.**(-1. / gam) - 1) * gamma_gam)
    # eq. 18
    alp = m0 - bet * gamma_gam

    return alp, bet, gam


def fit_basquin_goodman(cyc_data,
                        m_start=11.,
                        Rt_start=40.):
    '''
    Fits m and Rt to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :return: m_fit
    :return: Rt_fit
    '''

    initial_guess = [m_start, Rt_start]

    def objective(params):
        m, Rt = params  # multiple design variables

        crits = []
        for smax_actual, R, Nactual in zip(
                cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']]):
            if not R == 1:
                N = Nallow_basquin_goodman(smax_actual, R, m, Rt, M=1)
            else:
                N = 1
            crit1 = (np.log10(N) - np.log10(Nactual))**2
            crits.append(crit1)

            include_sigma_max_fit = True
            if include_sigma_max_fit:
                smax = smax_basquin_goodman(Nactual, R, m, Rt, M=1)
                crit2 = (np.log10(smax) - np.log10(smax_actual))**2
                crits.append(crit2)

        critsum = np.sum(np.array(crits))  # objective function
        return critsum
    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
        # print(fitted_params)
    else:
        raise ValueError(result.message)

    m_fit, Rt_fit = fitted_params

    return m_fit, Rt_fit


def fit_basquin_goodman_weibull(cyc_data,
                                m_start=11.,
                                Rt_start=40.):
    '''
    Fits m and Rt to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :return: m_fit
    :return: Rt_fit
    '''

    initial_guess = [m_start, Rt_start]

    def objective(params):
        m, Rt = params  # multiple design variables
        M = 1  # constants
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_smax_basquin_goodman(
                smax_actual, Nactual, R, m, Rt, M)

        global alp, bet, gam
        alp, bet, gam = pwm_weibull(xs_smax)

        crits = []

        for i, (smax_actual, R, Nactual) in enumerate(zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                                                          cyc_data['cyc_ratios'][cyc_data['grps']],
                                                          cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            smax_50 = smax_basquin_goodman_weibull(
                Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                p=p, alpha=alp, beta=bet, gamma=gam)

            crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2

            crits.append(crit1)

        critsum = np.sum(np.array(crits))  # objective function
        return critsum

        '''
        for smax_actual, R, Nactual in zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                                           cyc_data['cyc_ratios'][cyc_data['grps']],
                                           cyc_data['cyc_cycles'][cyc_data['grps']]):
            if not R == 1:
                N = Nallow_basquin_goodman(smax_actual, R, m, Rt, M=1)
            else:
                N = 1
            crit1 = (np.log10(N) - np.log10(Nactual))**2
            crits.append(crit1)

            include_sigma_max_fit = True
            if include_sigma_max_fit:
                smax = smax_basquin_goodman(Nactual, R, m, Rt, M=1)
                crit2 = (np.log10(smax) - np.log10(smax_actual))**2
                crits.append(crit2)

        critsum = np.sum(np.array(crits))  # objective function
        return critsum
        '''
    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
        # print(fitted_params)
    else:
        raise ValueError(result.message)

    m_fit, Rt_fit = fitted_params

    return m_fit, Rt_fit, alp, bet, gam


def fit_stuessi_goodman_weibull(cyc_data,
                                m_start=11.,
                                Rt_start=70.,
                                Re_start=40.,
                                Na_start=1E3,
                                n0=0.):
    '''
    Fits m, Rt, Re, Na and Weibull parameters to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :param: Re_start: float start value
    :param: Na_start: float start value
    :return: m_fit
    :return: Rt_fit
    :return: Re_fit
    :return: Na_fit
    :return: alp
    :return: bet
    :return: gam
    Note: a fitted curve can be obtained with:
    smax = smax_stuessi_goodman_weibull(n, R=R, m=m_fit, Rt=Rt_fit, M=M,
                Re=Re_fit, Na=Na_fit, p=p, alpha=alp, beta=bet, gamma=gam)
    '''
    initial_guess = [m_start, Rt_start, Re_start, Na_start]

    def objective(params):
        m, Rt, Re, Na = params  # multiple design variables
        M = 1  # constants
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_smax_stuessi_goodman(
                smax_actual, Nactual, R, m, Rt, M, Re, Na, n0)

        global alp, bet, gam
        alp, bet, gam = pwm_weibull(xs_smax)

        crits = []
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            smax_50 = smax_stuessi_goodman_weibull(
                Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                Re_fit=Re, Na=Na,
                p=p, alpha=alp, beta=bet, gamma=gam, n0=n0)

            crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2

            crits.append(crit1)

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

    m_fit, Rt_fit, Re_fit, Na_fit = fitted_params

    return m_fit, Rt_fit, Re_fit, Na_fit, alp, bet, gam
