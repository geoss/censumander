'''
Functions to compute various intermediate statistics.
'''

__author__ = "David C. Folch <dfolch@gmail.com>, Seth E. Spielman <seth.spielman@colorado.edu>"


import numpy as np
import time
import shapely.geometry
import shapely.ops
import bottleneck as bn


# localize a bunch of functions
nansum = bn.nansum
nanvar = bn.nanvar
ss = bn.ss
np_sqrt = np.sqrt
np_isfinite = np.isfinite
np_bitwise_not = np.bitwise_not
np_array = np.array
np_square = np.square
np_arange = np.arange
np_any = np.any
np_int32 = np.int32
np_isnan = np.isnan
np_isinf = np.isinf


def list_copy(l):
    # a fast function to copy a list
    return [i[:] for i in l]

def sde_count(sdes):
    '''
    Standard error calculation for count data.
    '''
    return np_sqrt((np_square(sdes)).sum())

def sum_squares(region, target_est):
    '''
    Calculate the univariate or multivariate sum of squared differences.
    '''
    ests = target_est.take(region, axis=0)
    return nansum(nanvar(ests, axis=0) * ests.shape[0], axis=None)

def compactness_global(regions, shp):
    '''
    Calculate compactness on each region.
    In the mathematics literature called "isoperimetric quotient" (see
    http://en.wikipedia.org/wiki/Isoperimetric_quotient) or in political
    science see Polsby and Popper (1991).
    '''
    quotients = np.ones(len(regions)) * -999.0
    for index, region in enumerate(regions):
        shapes = [shp[area] for area in region]
        o = []
        for shape in shapes:
            o.append(shapely.geometry.asShape(shape))
        res = shapely.ops.unary_union(o)
        quotients[index] = (4 * np.pi * res.area) / (res.length**2)
    return quotients


def get_est_sde_count(region, target_parts):
    reg = np_array(region, copy=False, dtype=np_int32)
    sdes = target_parts['target_sde_count'][reg] 
    est = target_parts['target_est_count'][reg].sum(0)
    sde = np_sqrt((np_square(sdes)).sum(0))
    return est, sde

def get_est_sde_prop(region, target_parts):
    target_est = target_parts['target_est_prop']
    target_sde = target_parts['target_sde_prop']
    cols = target_est.shape[1]
    reg = np_array(region, copy=False, dtype=np_int32)
    num_indexes = np_arange(0, cols, 2)
    den_indexes = num_indexes + 1    # does not work with numba
    target_sde_reg = target_sde.take(reg, axis=0)
    sde_num = np_sqrt(ss(target_sde_reg.take(num_indexes, axis=1), axis=0))
    sde_den = np_sqrt(ss(target_sde_reg.take(den_indexes, axis=1), axis=0))
    target_est_reg = target_est.take(reg, axis=0)
    est_num = nansum(target_est_reg.take(num_indexes, axis=1), axis=0)
    est_den = nansum(target_est_reg.take(den_indexes, axis=1), axis=0)
    est = est_num / est_den
    inside_sqrt = sde_num**2 - ((est**2) * sde_den**2)
    problems = inside_sqrt <= 0
    if np_any(problems):
        # deal with the (rare) case of negatives inside the square root
        inside_sqrt_alt = sde_num**2 + ((est**2) * sde_den**2)
        inside_sqrt[problems] = inside_sqrt_alt[problems]
    sde = np_sqrt(inside_sqrt) / est_den
    return est, sde

def get_est_sde_ratio(region, target_parts):
    target_est = target_parts['target_est_ratio']
    target_sde = target_parts['target_sde_ratio']
    cols = target_est.shape[1]
    reg = np_array(region, copy=False, dtype=np_int32)
    num_indexes = np_arange(0, cols, 2)
    den_indexes = np_arange(1, cols, 2)
    target_sde_reg = target_sde.take(reg, axis=0)
    sde_num = np_sqrt(ss(target_sde_reg.take(num_indexes, axis=1), axis=0))
    sde_den = np_sqrt(ss(target_sde_reg.take(den_indexes, axis=1), axis=0))
    
    target_est_reg = target_est.take(reg, axis=0)
    est_num = nansum(target_est_reg.take(num_indexes, axis=1), axis=0)
    est_den = nansum(target_est_reg.take(den_indexes, axis=1), axis=0)

    est = est_num / est_den
    inside_sqrt = sde_num**2 + ((est**2) * sde_den**2)

    sde = np_sqrt(inside_sqrt) / est_den
    return est, sde

def get_cv(est, sde, cv_exclude_type):
    '''
    we take a somewhat aggressive approach to problematic CV values:
    -if the estimate for any particular variable is very small,
     say less than 5% (e.g. unemployment rate of less than 5%) then we
     force that CV to zero, essentially guaranteeing that its magnitude
     will be less than any user defined threshold
    -if the estimate for any particular variable is NAN, then we
     force that CV to zero, essentially guaranteeing that its magnitude
     will be less than any user defined threshold, we assume this can
     only happen in the 0/0 case for the given input data (this may need
     to be revisited to increase generality)
    -if the CV for any particular variable is Inf (which I don't think
     is possible given the previous two rules), then it will be forced to
     1.0, which should fail most user's criteria
    '''
    cv = sde / est
    cv[est < cv_exclude_type] = 0.0
    cv[np_isnan(est)] = 0.0
    cv[np_isinf(cv)] = 1.0
    return cv.tolist()



def get_mv_cv(region, target_parts, cv_exclude):
    '''
    coefficient of variation for multiple data sets
    NOTE: need to investigate refactoring this for more speed
    '''
    # IMPORTANT: maintain the order of count then proportion then ratio
    cvs = []
    # compute coefficient of variation for each count variable
    if target_parts['target_est_count'] is not None:
        est, sde = get_est_sde_count(region, target_parts)
        cvs.extend(get_cv(est, sde, cv_exclude[0]))

    # compute coefficient of variation for each proportion variable
    if target_parts['target_est_prop'] is not None:
        est, sde = get_est_sde_prop(region, target_parts)
        cvs.extend(get_cv(est, sde, cv_exclude[1]))
    
    # compute coefficient of variation for each ratio variable
    if target_parts['target_est_ratio'] is not None:
        est, sde = get_est_sde_ratio(region, target_parts)
        cvs.extend(get_cv(est, sde, cv_exclude[2]))
    return cvs


