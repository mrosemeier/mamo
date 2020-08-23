import numpy as np
import scipy.optimize as optimize
from scipy.special import gamma


def smax_basquin_goodman(n, R=-1, m=10, R_t=1.0, M=1):
    '''
    Max stress for given cycle number of an SN curve according to Basquin-Goodman
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: R_t: float static strength
    :param: M: mean stress sensitivity
    :return: smax: float allowable maximum stress
    '''
    e_max = 2. / ((1. - R) * n**(1. / m) + M * (R + 1))
    return R_t * e_max


def N_allow_basquin_goodman(smax, R, m, R_t, M):
    '''
    Allowable cycles for given ma xstress of an SN curve according to Basquin-Goodman
    :param: smax: float max stress
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: R_t: float static strength
    :param: M: mean stress sensitivity
    :return: N_allow: float allowable cycles
    '''
    e_max = smax / R_t
    return ((e_max * M * (R + 1.) - 2.) / (e_max * (R - 1.)))**m


def _b(m, R_d, R_t):
    '''
    Transforms Basquin's m into Stuessi's b
    (1) Derive Basquin for s: dn/ds(s) = -(N_a*m*(s/R_a)**(-m))/s
    (2) Derive Stuessi for s: dn/ds(s) = -(N_a*b*(R_d-R_t)*((-R_d+s)/(R_t-s))**(-b))/((R_d-s)*(R_t-s))
    (3) Set R_a = 0.5*(R_d+R_t)
    (4) Set (1)=(2) with (3) and solve for b
    '''
    return (m * (-R_d + R_t)) / (2 * (R_d + R_t))


def smax_stuessi_goodman(n, R=-1, m=10, R_t=1.0, M=1, R_d=30, N_a=1E3):
    '''
    Max stress of an SN curve according to Stuessi-Goodman
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at N_a for R=-1
    :param: R_t: float static strength
    :param: M: mean stress sensitivity
    :param: R_d: float endurance limit for R=-1
    :param: N_a: float turning point
    :return: smax: float allowable maximum stress
    '''
    b = _b(m, R_d, R_t)
    return (2. * R_t * (R_d * (n / N_a)**(1. / b) + R_t)) / \
        (-R_t * (-1 + (n / N_a)**(1. / b) * (R - 1) - R *
                 (M - 1.) - M) + R_d * (n / N_a)**(1. / b) * (1. + R) * M)


def xrand_smax_stuessi_goodman(smax_i, N_i, R_i, m_fit, R_t_fit, M_fit, R_d_fit, N_a_fit):
    ''' Random variable x using smax
    '''
    return smax_i - smax_stuessi_goodman(N_i, R_i, m_fit, R_t_fit, M_fit, R_d_fit, N_a_fit)


def x_weibull(p, alp, bet, gam):
    ''' 
    Obains x value of a Weibull dsitribution for a given probability p
    (1) solve p = 1. - np.exp(-((x - alpha) / beta)**gamma) for x if beta>0
    and 
    (2) solve p = np.exp(-((x - alpha) / beta)**gamma) for x if beta<0
    Note: A negative beta flips the Weibull curve about x-axis
    :param: p: float probability
    :param: alp: float Weibull parameter
    :param: bet: float Weibull parameter
    :param: gam: float Weibull parameter
    :return: x: float value
    '''
    if bet < 0:
        x = alp + bet * (-np.log(p))**(1. / gam)
    else:
        x = alp + bet * (-np.log(1. - p))**(1. / gam)
    return x


def smax_stuessi_goodman_weibull(n, R, m, R_t, M, R_d, N_a, p, alpha, beta, gamma):
    '''
    Max stress for given cycle number and probability of an SN curve according to Stuessi-Goodman
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at N_a for R=-1
    :param: R_t: float static strength
    :param: M: mean stress sensitivity
    :param: R_d: float endurance limit for R=-1
    :param: N_a: float turning point
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: smax: float allowable maximum stress
    '''
    return x_weibull(p, alpha, beta, gamma) + smax_stuessi_goodman(n, R, m, R_t, M, R_d, N_a)


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
                        R_t_start=40.):
    '''
    Fits m and R_t to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: R_t_start: float start value
    :return: m_fit
    :return: R_t_fit
    '''

    initial_guess = [m_start, R_t_start]

    def objective(params):
        m, R_t = params  # multiple design variables

        crits = []
        for smax_actual, R, N_actual in zip(cyc_data['cyc_smax'][cyc_data['grps']],
                                            cyc_data['cyc_ratios'][cyc_data['grps']],
                                            cyc_data['cyc_cycles'][cyc_data['grps']]):
            if not R == 1:
                N = N_allow_basquin_goodman(smax_actual, R, m, R_t, M=1)
            else:
                N = 1
            crit1 = (np.log10(N) - np.log10(N_actual))**2
            crits.append(crit1)

            include_sigma_max_fit = True
            if include_sigma_max_fit:
                smax = smax_basquin_goodman(N_actual, R, m, R_t, M=1)
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

    m_fit, R_t_fit = fitted_params

    return m_fit, R_t_fit


def fit_stuessi_goodman_weibull(cyc_data,
                                m_start=11.,
                                R_t_start=70.,
                                R_d_start=40.,
                                N_a_start=1E3):
    '''
    Fits m, R_t, R_d, N_a and Weibull parameters to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: R_t_start: float start value
    :param: R_d_start: float start value
    :param: N_a_start: float start value
    :return: m_fit
    :return: R_t_fit
    :return: R_d_fit
    :return: N_a_fit
    :return: alp
    :return: bet
    :return: gam
    Note: a fitted curve can be obtained with:
    smax = smax_stuessi_goodman_weibull(n, R=R, m=m_fit, R_t=R_t_fit, M=M,
                R_d=R_d_fit, N_a=N_a_fit, p=p, alpha=alp, beta=bet, gamma=gam)
    '''
    initial_guess = [m_start, R_t_start, R_d_start, N_a_start]

    def objective(params):
        m, R_t, R_d, N_a = params  # multiple design variables
        M = 1  # constants
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, N_actual) in enumerate(zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                                                           cyc_data['cyc_ratios'][cyc_data['grps']],
                                                           cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_smax_stuessi_goodman(
                smax_actual, N_actual, R, m, R_t, M, R_d, N_a)

        global alp, bet, gam
        alp, bet, gam = pwm_weibull(xs_smax)

        crits = []
        for i, (smax_actual, R, N_actual) in enumerate(zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                                                           cyc_data['cyc_ratios'][cyc_data['grps']],
                                                           cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            smax_50 = smax_stuessi_goodman_weibull(
                N_actual, R=R, m=m, R_t=R_t, M=M,
                R_d=R_d, N_a=N_a,
                p=p, alpha=alp, beta=bet, gamma=gam)

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

    m_fit, R_t_fit, R_d_fit, N_a_fit = fitted_params

    return m_fit, R_t_fit, R_d_fit, N_a_fit, alp, bet, gam
