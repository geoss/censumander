'''
A regionalization scheme for ACS data.

This algorthm is an implimnetion of Duque et al (2012) in the sense that the
number of spatially contiguous regions created is determined within the
algorithm, i.e. not selected a priori.  The details and motivations for this
specific approach taken here can be found in  Spielman and Folch (2014) and
Folch and Spielman (2014).
'''

__author__ = "David C. Folch <dfolch@gmail.com>, Seth E. Spielman <seth.spielman@colorado.edu>"


import pysal as ps
import numpy as np
import time
import copy
import scipy.spatial
from scipy.stats.mstats import zscore as ZSCORE
import mdp as MDP  # causing sklearn deprication warning
import pandas as pd
import base_construction as BASE
import local_search as LOCAL
import utils as UTILS



class ACS_Regions:
    '''


    Parameters
    ----------
    w                : w
                       pysal weights object

    target_est_count : array or None
                       nxc array, where c is the number of count variables;
                       reliability test is applied to this array; if None then
                       it is ignored
    target_moe_count : array or None
                       nxc array of margins of error associated with
                       target_est_count; if None then it is ignored
    target_th_count  : array or None
                       array of length c giving the maximum coefficient of
                       variation associated with each vector in
                       target_est_count (decimal values between 0.0 and 1.0);
                       ignored if target_th_all>0

    target_est_prop  : array or None
                       nx2f array, where f is the number of proportion
                       variables; for each f the array must contain the
                       numerator and denominator in the form [num1, den1,
                       num2, den2,..., numf, denf]; reliability test is
                       applied to this array; if None then it is ignored
    target_moe_prop  : array or None
                       nx2f array of margins of error associated with
                       target_est_prop, must be of the same form as
                       target_est_prop; if None then it is ignored
    target_th_prop   : array or None
                       array of length f giving the maximum coefficient of
                       variation associated with each proportion variable
                       represented in target_est_prop (decimal values 
                       between 0.0 and 1.0); ignored if target_th_all>0

    target_est_ratio : array or None
                       nx2r array, where r is the number of ratio     
                       variables; for each r the array must contain the
                       numerator and denominator in the form [num1, den1,
                       num2, den2,..., numr, denr]; reliability test is
                       applied to this array; if None then it is ignored
    target_moe_ratio : array or None
                       nx2r array of margins of error associated with
                       target_est_ratio, must be of the same form as
                       target_est_ratio; if None then it is ignored
    target_th_ratio  : array or None
                       array of length r giving the maximum coefficient of
                       variation associated with each proportion variable
                       represented in target_est_ratio (decimal values 
                       between 0.0 and 1.0); ignored if target_th_all>0

    target_th_all    : float or int
                       single maximum coefficient of variation for a feasible
                       region applied to all target variables (a decimal
                       value between 0.0 and 1.0); if >0 will override all
                       other threshold values

    count_est        : array
                       variable to count (e.g. population, housing units)
    count_th_min     : int or float
                       minimum sum of count_est for a feasible region
                       (see count_th_max)
    count_th_max     : int or float
                       maximum sum of count_est for a feasible region
                       (see count_th_min)

    exclude          : list
                       observation IDs to exclude from the regionalization; 
                       assumed to be in the same format as the IDs in the W
                       object
    auto_exclude     : float
                       If the share of remaining areas from the base_solution
                       is less than auto_exclude, then add these remaining areas
                       to the exclude list for swapping phase; default is set
                       to zero meaning that the presence of any unassigned areas
                       will result in an infeasible solution
    base_solutions   : integer
                       number of initial solutions to generate
    zscore           : boolean
                       If True standardize all data using z-scores; if False
                       use data as passed in. This allows the user to convert
                       their data outside this function by any means, and then
                       pass it to this function using zscore=False.
    pca              : boolean
                       If True convert the input data using principle
                       components analysis. The resulting components are then
                       used for computing sum of squared deviations (SSD).
                       Note that each component's contribution to global SSD is
                       weighted by that component's share of total variation.
                       If False do not transform the variables using PCA.
    local_improvement: boolean
                       if true, then run the local improvement stage of the
                       algorithm
    local_params     : dictionary
                       Optional parameter settings for the local search stage.
                       The following are all integers that are multiplied by
                       the number of regions determined within the algorithm
                       'max_swaps': maximum number of improvements allowed
                       'max_chain': maximum number of consecutive
                                    non-improving moves
                       'max_tabu': maximum length of tabu list
                       'change_thresh' (float between 0 and 1): if the percent
                                       change in the objective function between
                                       the current move and the 10th prior
                                       move is less than change_thresh, then 
                                       stop the algorithm.
    compactness      : list of objects or None
                       list of objects compatible with shapely.geometry.asShape(),
                       this list can be generated by pysal.open('my_shp.shp');
                       if list present then compute compactness for each
                       region; if None, then do nothing
    points           : None, array, KDTree, shp
                       If None use a random approach for choosing seeds for
                       region building; if KDTree (see pysal.common.KDTree())
                       then use the Vickrey (1961) quasi-random approach for
                       choosing regions seeds; if nx2 numpy array of points,
                       then build the KDTree and run the Vickrey approach; if
                       shp (object created from
                       pysal.open('my_shapefile.shp')), then find polygon
                       centroids, build KDTree and run the Vickrey approach
    anchor           : None or int
                       If int, then use this ID as the anchor in the Vickery
                       (1961) approach for finding the base solution (this
                       overrides the value in base_solutions); if None then
                       randomly choose the anchor IDs; ignored if points=None
    cardinality      : boolean
                       If True, then areas with fewer neighbors will be the
                       first seed candidates.  If False, don't consider
                       neighbor structure when ordering potential region
                       seeds.  Since many areas have the same number of
                       neighbors, ties are broken randomly if points=None,
                       otherwise using the Vickrey approach
    cv_exclude_count : int
                       If the estimate for any count attribute is below this
                       level, then the CV target threshold (see above) will be
                       ignored for this attribute
    cv_exclude_prop  : float
                       If the estimate for any proportion attribute is below this
                       level, then the CV target threshold (see above) will be
                       ignored for this attribute
    cv_exclude_ratio : float
                       If the estimate for any ratio attribute is below this
                       level, then the CV target threshold (see above) will be
                       ignored for this attribute
                     
    Attributes          
    ----------          
    exit             : string
                       reason the algorithm ended
    time             : dict
                       'prep':algorithm setup; 'base':initialization phase;
                       'base_wrapup':initialization phase results collection;
                       'local':improvement phase; 'local_wrapup':improvement 
                       phase results collection; 'wrapup':algorithm wrapup;
                       'total':total time; all times measured in seconds
    p                : int
                       number of regions formed
    enclaves         : int
                       number of enclaves that had to be assigned during
                       initialization phase
    regions          : list of lists
                       p length list, where each subordinate list contains the
                       area IDs for that region; row number matches region ID
    region_ids       : list
                       list of length n, linking area to its region ID; order
                       follows that of input data
    ssds             : pandas dataframe
                       p x 4 dataframe; columns are regionID, start_ssd,
                       end_ssd, ssd_improvement
    compactness      : pandas dataframe
                       p x 4 dataframe; columns are regionID, start_compactness,
                       end_compactness, compactness_improvement
    ests_region      : pandas dataframe 
                       region estimates for all attributes (ordered by region ID)
    moes_region      : pandas dataframe
                       region MOEs for all attributes (ordered by region ID)
    cvs_region       : pandas dataframe
                       region CVs for all attributes (ordered by region ID)
    ests_area        : pandas dataframe
                       area estimates for all attributes (order matches input data)
    moes_area        : pandas dataframe
                       area MOEs for all attributes (order matches input data)
    cvs_area         : pandas dataframe
                       area CVs for all attributes (order matches input data)
    counts_region    : pandas dataframe
                       region count variable (included if user supplies count_est)
    counts_area      : pandas dataframe
                       area count variable (included if user supplies count_est)
    problem_ids      : list
                       IDs that could not be joined to a region
    '''

    def __init__(self, w, target_est_count=None, target_moe_count=None, target_th_count=None,\
                    target_est_prop=None, target_moe_prop=None, target_th_prop=None,\
                    target_est_ratio=None, target_moe_ratio=None, target_th_ratio=None,\
                    target_th_all=None, count_est=None, count_th_min=None, count_th_max=None,\
                    exclude=None, auto_exclude=0, base_solutions=100,\
                    zscore=True, pca=True, local_improvement=True, local_params=None,\
                    compactness=None, points=None, anchor=None, cardinality=False,\
                    cv_exclude_count=0, cv_exclude_prop=0, cv_exclude_ratio=0):

        time1 = time.time()
        time_output = {'prep':0, 'base':0, 'base_wrapup':0, 'local':0,
                       'local_wrapup':0, 'wrapup':0, 'total':0}
        # convert arbitrary IDs in W object to integers
        id2i = w.id2i
        neighbors = {id2i[key]:[id2i[neigh] for neigh in w.neighbors[key]] for key in w.id_order}
        w = ps.W(neighbors)
        
        # build KDTree for use in finding base solution
        if issubclass(type(points), scipy.spatial.KDTree):
            kd = points
            points = kd.data
        elif type(points).__name__ == 'ndarray':
            kd = ps.common.KDTree(points)
        elif issubclass(type(points), ps.core.IOHandlers.pyShpIO.PurePyShpWrapper):
            #loop to find centroids, need to be sure order matches W and data
            centroids = []
            for i in points:
                centroids.append(i.centroid)
            kd = ps.common.KDTree(centroids)
            points = kd.data
        elif points is None:
            kd = None
        else:
            raise Exception, 'Unsupported type passed to points'


        # dictionary allowing multivariate and univariate flexibility
        target_parts = {'target_est_count':target_est_count,\
                        'target_est_prop':target_est_prop,\
                        'target_est_ratio':target_est_ratio,\
                        'target_sde_count':target_moe_count ,\
                        'target_sde_prop':target_moe_prop,\
                        'target_sde_ratio':target_moe_ratio}

        # setup the holder for the variables to minimize; later we will put all
        # the count, ratio and proportion variables into this array.
        # Also, convert MOEs to standard errors when appropriate
        total_vars = 0
        rows = 0
        if target_est_count is not None:
            rows, cols = target_est_count.shape
            total_vars += cols
            target_parts['target_est_count'] = target_est_count * 1.0
            target_parts['target_sde_count'] = target_moe_count / 1.645
        if target_est_prop is not None:
            rows, cols = target_est_prop.shape
            total_vars += cols/2
            target_parts['target_est_prop'] = target_est_prop * 1.0
            target_parts['target_sde_prop'] = target_moe_prop / 1.645
        if target_est_ratio is not None:
            rows, cols = target_est_ratio.shape
            total_vars += cols/2
            target_parts['target_est_ratio'] = target_est_ratio * 1.0
            target_parts['target_sde_ratio'] = target_moe_ratio / 1.645

        if total_vars == 0:
            target_est = None
            print 'warning: optimization steps will not be run since no target_est variables provided'
        else:
            target_est = np.ones((rows, total_vars)) * -999

        # organize and check the input data; prep data for actual computations
        position = 0
        target_th = []
        # IMPORTANT: maintain the order of count then proportion then ratio
        if target_est_count is not None:
            target_est, target_th, position = mv_data_prep(target_est_count,\
                                              target_th_count, target_th_all,\
                                              target_est, target_th, position,\
                                              scale=1, ratio=False)
        if target_est_prop is not None:
            target_est, target_th, position = mv_data_prep(target_est_prop,\
                                              target_th_prop, target_th_all,\
                                              target_est, target_th, position,\
                                              scale=2, ratio=False)
        if target_est_ratio is not None:
            target_est, target_th, position = mv_data_prep(target_est_ratio,\
                                              target_th_ratio, target_th_all,\
                                              target_est, target_th, position,\
                                              scale=2, ratio=True)
        target_th = np.array(target_th)

        
        # compute zscores
        # NOTE: zscores computed using all data, i.e. we do not screen out
        #       observations in the exclude list.
        if zscore and target_est is not None:
            if pca:
                # Python does not currently have a widely used tool for
                # computing PCA with missing values. In principle, 
                # NIPALS (Nonlinear Iterative Partial Least Squares)
                # can accommodate missing values, but the implementation in MDP
                # 3.4 will return a matrix of NAN values if there is an NAN
                # value in the input data. 
                # http://sourceforge.net/p/mdp-toolkit/mailman/mdp-toolkit-users/?viewmonth=201111
                # http://stats.stackexchange.com/questions/35561/imputation-of-missing-values-for-pca
                # Therefore, we impute the missing values when the user
                # requests PCA; compute the z-scores on the imputed data; and
                # then pass this on to the PCA step. 
                # The imputation replaces a missing value with the average of
                # its neighbors (i.e., its spatial lag). If missing values
                # remain (due to missing values in a missing value's neighbor
                # set), then that value is replaced by the column average.
                w_standardized = copy.deepcopy(w)
                w_standardized.transform = 'r'
                target_est_lag = ps.lag_spatial(w_standardized, target_est)
                # replace troublemakers with their spatial lag
                trouble = np.isfinite(target_est)
                trouble = np.bitwise_not(trouble)
                target_est[trouble] = target_est_lag[trouble]
                del target_est_lag
                del trouble
            # Pandas ignores missing values by default, so we can
            # compute the z-score and retain the missing values
            target_est = pd.DataFrame(target_est)
            target_est = (target_est - target_est.mean(axis=0)) / target_est.std(axis=0)
            target_est = target_est.values
            if pca:
                # For the PCA case we need to replace any remaining missing
                # values with their column average. Since we now have z-scores,
                # we know that the average of every column is zero.
                # If it's not the PCA case, then we can leave the missing
                # values in as they will be ignored down the line.
                if np.isfinite(target_est.sum()) == False:
                    trouble = np.isfinite(target_est)
                    trouble = np.bitwise_not(trouble)
                    target_est[trouble] = 0.
                    del trouble

        # run principle components on target data (skip PCA if pca=False)
        # NOTE: matplotlib has deprecated PCA function, also it only uses SVD 
        #       which can get tripped up by bad data
        # NOTE: the logic here is to first identify the principle components and
        #       then weight each component in preparation for future SSD
        #       computations; we weight the data here so that we don't need to 
        #       weight the data each time the SSD is computed; in effect we want
        #       to compute the SSD on each raw component and then weight that
        #       component's contribution to the total SSD by the component's share
        #       of total variance explained, since the SSD computation has a
        #       squared term we can take the square root of the data now and then
        #       not have to weight it later
        # NOTE: PCA computed using all data, i.e. we do not screen out
        #       observations in the exclude list.
        if pca and target_est is not None:
            try:
                # eigenvector approach
                pca_node = MDP.nodes.PCANode()
                target_est = pca_node.execute(target_est)  # get principle components
            except:
                try:
                    # singular value decomposition approach
                    pca_node = MDP.nodes.PCANode(svd=True)
                    target_est = pca_node.execute(target_est)  # get principle components
                except:
                    # NIPALS would be a better approach than imputing
                    # missing values entirely, but MDP 3.4 does not handle
                    # missing values. Leaving this code as a place holder in
                    # case MDP is updated later.
                    ###pca_node = MDP.nodes.NIPALSNode()
                    ###target_est = pca_node.execute(target_est)  # get principle components
                    raise Exception, "PCA not possible given input data and settings. Set zscore=True to automatically impute missing values or address missing values in advance."

            pca_variance = np.sqrt(pca_node.d / pca_node.total_variance)
            target_est = target_est * pca_variance  # weighting for SSD

        # NOTE: the target_est variable is passed to the SSD function, and the
        #       target_parts variable is passed to the feasibility test function

        # set the appropriate objective function plan 
        build_region, enclave_test, local_test = function_picker(count_est,\
                                    count_th_min, count_th_max, target_th_count,\
                                    target_th_prop, target_th_ratio, target_th_all)

        # setup the CV computation
        get_cv = UTILS.get_mv_cv
        cv_exclude = [cv_exclude_count, cv_exclude_prop, cv_exclude_ratio]

        # setup areas to be excluded from computations
        if exclude:
            exclude = [id2i[j] for j in exclude]
            original_exclude = exclude[:]  # in integer ID form
        else:
            original_exclude = []
        # might consider an automated process to drop observations where
        # count_est=0; at this time the user would be expected to add these
        # observations to the exclude list


        time2 = time.time()
        time_output['prep'] = time2 - time1
        # find the feasible solution with the most number of regions
        regions, id2region, exclude, enclaves = BASE.base_region_iterator(\
                             w, count_th_min, count_th_max, count_est, target_th, target_est,\
                             exclude, auto_exclude, get_cv, base_solutions,\
                             target_parts, build_region, enclave_test, kd, points,
                             anchor, cardinality, cv_exclude)

        time3 = time.time()
        time_output['base'] = time3 - time2
        problem_ids = list(set(exclude).difference(original_exclude))
        if id2region == False:
            # Infeasible base run
            exit = "no feasible solution"
            time3a = time4 = time4a = time.time()
        else:
            if target_est is not None:
                # only compute SSDs if there are target_est variables
                start_ssds = np.array([UTILS.sum_squares(region, target_est) for region in regions])
            else:
                start_ssds = np.ones(len(regions)) * -999.0

            if compactness:
                # capture compactness from base solution
                start_compactness = UTILS.compactness_global(regions, compactness)

            if local_improvement and len(regions)>1:
                # only run the local improvement if the appropriate flag is set
                # (local_improvement=True) and if there is more then one region to
                # swap areas between
                # swap areas along region borders that improve SSD
                time3a = time.time()
                regions, id2region, exit = \
                              LOCAL.local_search(regions, id2region, w, count_th_min, count_th_max,\
                                                 count_est, target_th, target_parts,\
                                                 target_est, exclude, get_cv,\
                                                 local_test, local_params, cv_exclude)
                time4 = time.time()
                # collect stats on SSD for each region
                end_ssds = np.array([UTILS.sum_squares(region, target_est) for region in regions])
                ssd_improvement = (end_ssds - start_ssds) / start_ssds
                ssd_improvement[np.isnan(ssd_improvement)] = 0.0  # makes singleton regions have 0 improvement
                ssds = np.vstack((start_ssds, end_ssds, ssd_improvement)).T
                if compactness:
                    # capture compactness from final solution
                    end_compactness = UTILS.compactness_global(regions, compactness)
                    compact_change = \
                        (end_compactness - start_compactness) / start_compactness
                    compacts = np.vstack((start_compactness, end_compactness, compact_change)).T
                else:
                    compacts = np.ones((len(regions),3)) * -999.0
                time4a = time.time()
            else:
                time3a = time4 = time.time()
                # capture start SSDs and compactness, insert -999 for "improvements"
                ssds = np.vstack((start_ssds, np.ones(start_ssds.shape)*-999,\
                                              np.ones(start_ssds.shape)*-999)).T
                if compactness:
                    compacts = np.vstack((start_compactness, np.ones(start_compactness.shape)*-999,\
                                                             np.ones(start_compactness.shape)*-999)).T
                else:
                    compacts = np.ones((len(regions),3)) * -999.0
                exit = 'no local improvement'
                print "Did not run local improvement"
                time4a = time.time()

        time_output['base_wrapup'] = time3a - time3
        time_output['local'] = time4 - time3a
        time_output['local_wrapup'] = time4a - time4


        ####################
        # process regionalization results for user output
        ####################

        # setup header for the pandas dataframes (estimates, MOEs, CVs)
        header = []
        if target_est_count is not None:
            if 'pandas' in str(type(target_est_count)):
                header.extend(target_est_count.columns.tolist())
            else:
                header.extend(['count_var'+str(i) for i in range(target_est_count.shape[1])])
        if target_est_prop is not None:
            if 'pandas' in str(type(target_est_prop)):
                header.extend(target_est_count.prop.tolist())
            else:
                header.extend(['prop_var'+str(i) for i in range(target_est_prop.shape[1]/2)])
        if target_est_ratio is not None:
            if 'pandas' in str(type(target_est_ratio)):
                header.extend(target_est_ratio.columns.tolist())
            else:
                header.extend(['ratio_var'+str(i) for i in range(target_est_ratio.shape[1]/2)])

        # initialize pandas dataframes (estimates, MOEs, CVs; regions and areas)
        regionID = pd.Index(range(len(regions)), name='regionID')
        ests_region = pd.DataFrame(index=regionID, columns=header)
        moes_region = pd.DataFrame(index=regionID, columns=header)
        cvs_region = pd.DataFrame(index=regionID, columns=header)
        areaID = pd.Index(range(w.n), name='areaID')
        ests_area = pd.DataFrame(index=areaID, columns=header)
        moes_area = pd.DataFrame(index=areaID, columns=header)
        cvs_area = pd.DataFrame(index=areaID, columns=header)

        # setup header and pandas dataframe (count variable, if applicable)
        header = ['count']
        if count_est is not None:
            if 'pandas' in str(type(count_est)):
                header = [count_est.columns[0]]
        counts_region = pd.DataFrame(index=range(len(regions)), columns=header)
        counts_area = pd.DataFrame(index=range(w.n), columns=header)

        # create SSD and compactness dataframes
        if id2region == False:
            # Infeasible base run
            ssds = None
            compacts = None
        else:
            ssds = pd.DataFrame(ssds, index=regionID, columns=['start_ssd',
                                             'end_ssd', 'ssd_improvement'])
            compacts = pd.DataFrame(compacts, index=regionID, columns=['start_compactness',
                                             'end_compactness', 'compactness_improvement'])

        # this one-dimensional list will contain the region IDs (ordered by area)
        ordered_region_ids = np.ones(w.n) * -9999

        for i, region in enumerate(regions):
            if count_est is not None:
                # get region totals for count variable
                counts_region.ix[i] = count_est[region].sum()
                for j in region:
                    counts_area.ix[j] = count_est[j]
            ests = []
            sdes = []
            if target_est_count is not None:
                # est, MOE and CV for count data
                est, sde = UTILS.get_est_sde_count(region, target_parts)
                est[np.isnan(est)] = 0.0   # clean up 0/0 case
                sde[np.isnan(sde)] = 0.0   # clean up 0/0 case
                ests.extend(est)
                sdes.extend(sde)
            if target_est_prop is not None:
                # est, MOE and CV for proportion data
                est, sde = UTILS.get_est_sde_prop(region, target_parts)
                est[np.isnan(est)] = 0.0   # clean up 0/0 case
                sde[np.isnan(sde)] = 0.0   # clean up 0/0 case
                ests.extend(est)
                sdes.extend(sde)
            if target_est_ratio is not None:
                # est, MOE and CV for ratio data
                est, sde = UTILS.get_est_sde_ratio(region, target_parts)
                est[np.isnan(est)] = 0.0   # clean up 0/0 case
                sde[np.isnan(sde)] = 0.0   # clean up 0/0 case
                ests.extend(est)
                sdes.extend(sde)
            ests_region, moes_region, cvs_region = wrapup_region(\
                                i, ests, sdes, target_parts,
                                ests_region, moes_region, cvs_region)
            ests_area, moes_area, cvs_area = wrapup_areas(\
                                region, target_parts,
                                ests_area, moes_area, cvs_area)
            ordered_region_ids[region] = i
        # set excluded areas to region ID -999
        ordered_region_ids[exclude] = -999
        time5 = time.time()
        time_output['wrapup'] = time5 - time4
        time_output['total'] = time5 - time1


        self.exit = exit
        self.time = time_output
        self.enclaves = enclaves
        self.p = len(regions)
        self.regions = regions
        self.region_ids = ordered_region_ids.tolist()
        self.ssds = ssds
        self.compactness = compacts
        self.ests_region = ests_region
        self.moes_region = moes_region
        self.cvs_region = cvs_region 
        self.ests_area = ests_area
        self.moes_area = moes_area
        self.cvs_area = cvs_area 
        self.counts_region = counts_region
        self.counts_area = counts_area
        self.problem_ids = problem_ids

def wrapup_region(i, ests, sdes, target_parts,
                  ests_region, moes_region, cvs_region):
    '''
    organize the output data
    '''
    ests = np.array(ests)
    sdes = np.array(sdes)
    ests_region.ix[i] = ests
    moes_region.ix[i] = sdes * 1.645
    cv = sdes / ests    # we bypass the cv_exclude rules for the output
    cv[np.isnan(ests)] = 0.0  # force CV to zero if estimate is zero
    cvs_region.ix[i] = cv
    return ests_region, moes_region, cvs_region

def wrapup_areas(region, target_parts,
                 ests_area, moes_area, cvs_area):
    '''
    organize the output data
    '''
    for j in region:
        ests = []
        sdes = []
        if target_parts['target_est_count'] is not None:
            est, sde = UTILS.get_est_sde_count([j], target_parts)
            ests.extend(est)
            sdes.extend(sde)
        if target_parts['target_est_prop'] is not None:
            est, sde = UTILS.get_est_sde_prop([j], target_parts)
            ests.extend(est)
            sdes.extend(sde)
        if target_parts['target_est_ratio'] is not None:
            est, sde = UTILS.get_est_sde_ratio([j], target_parts)
            ests.extend(est)
            sdes.extend(sde)
        ests = np.array(ests)
        sdes = np.array(sdes)
        ests_area.ix[j] = ests
        moes_area.ix[j] = sdes * 1.645
        cvs = sdes / ests    # we bypass the cv_exclude rules for the output
        cvs[np.isnan(ests)] = 0.0  # force CV to zero if estimate is zero
        cvs_area.ix[j] = cvs
    return ests_area, moes_area, cvs_area

def mv_data_prep(target_est_general, target_th_general, target_th_all,\
                 target_est, target_th, position, scale, ratio):
    '''
    Performs a number of data prep activities for the multivariate case. Some 
    consistency checks on the user provided data; organize the CV thresholds;
    compute proportion and ratio estimates. Note that there is complex
    handling of zeros in a divide (see below).
    '''
    rows, cols = target_est_general.shape
    if target_th_all:
        # set all the CV thresholds to the same value
        target_th.extend([target_th_all] * (cols/scale))
    else:
        # check to ensure there is one CV per variable
        if cols/scale != target_th_general.shape[0]:
            raise Exception, "input data does not conform"
        else:
            # set external CV thresholds
            target_th.extend(target_th_general)
    # compute the actual estimates for use in SSD computations
    if scale == 2:
        # ratio and proportion case
        target_combo_temp = np.ones([rows, cols/scale]) * -999.0
        col = 0
        for i in range(0, cols, 2):
            '''
            how we handle zeros in division (note: assumes we've previously replaced 
                                                   missing values with zeros)
            ratio:
            0 / 3.0 = np.nan  (illogical case: no home value but there are homes)
            0 / 0   = 0       (logical case: no home value and no homes)
            3.0 / 0 = np.nan  (illogical case: home value but no homes)
            proportion:
            0 / 3.0 = 0       (logical case: no white pop and some total pop)
            0 / 0   = 0       (logical case: no white pop and no total pop) 
            3.0 / 0 = np.nan  (illogical case: white pop with no total pop)
            '''
            num = (target_est_general[:,i]).copy() * 1.0
            den = (target_est_general[:,i+1]).copy() * 1.0
            num_zeros = num==0
            den_zeros = den==0
            both_zeros = num_zeros.copy()
            for index in range(both_zeros.shape[0]):
                if num_zeros[index]==True and den_zeros[index]==True:
                    both_zeros[index] = True
                else:
                    both_zeros[index] = False
            if ratio == True:
                num[num_zeros] = np.nan  # force np.nan  for  0 / 3.0  case
            den[den_zeros] = np.nan      # force np.nan  for  3.0 / 0  case
            num[both_zeros] = 0          # force zero    for    0 / 0  case
            den[both_zeros] = 99.        # force zero    for    0 / 0  case
            #target_combo_temp[:,col] = (target_est_general[:,i]*1.0) / target_est_general[:,i+1] 
            target_combo_temp[:,col] = num / den 
            col += 1
        # add variables to minimizer array
        target_est[:,position:position+cols/scale] = target_combo_temp
    else:
        # count data case
        target_est[:,position:position+cols/scale] = target_est_general
    position += cols/scale
    return target_est, target_th, position

def function_picker(count_est, count_th_min, count_th_max, target_th_count,\
                    target_th_prop=None, target_th_ratio=None, target_th_all=None):
    # set the appropriate objective function plan 
    if count_th_min is None and count_th_max is None:
        build_region = BASE.build_region_cv_only
        enclave_test = BASE.enclave_test_cv_only
        local_test = LOCAL.local_test_cv_only
    elif target_th_count is None and target_th_prop is None and\
                                     target_th_ratio is None and\
                                     target_th_all is None:
        build_region = BASE.build_region_count_only
        enclave_test = BASE.enclave_test_count_only
        local_test = LOCAL.local_test_count_only
    elif count_th_min > 0 and count_th_max is None:
        build_region = BASE.build_region_min_count
        enclave_test = BASE.enclave_test_cv_only # only need to test CV when adding areas
        local_test = LOCAL.local_test_min_count
    elif count_th_min is None and count_th_max > 0:
        build_region = BASE.build_region_max_count
        enclave_test = BASE.enclave_test_max_count
        local_test = LOCAL.local_test_max_count
    elif count_th_min > 0 and count_th_max > 0:
        build_region = BASE.build_region_min_max_count
        enclave_test = BASE.enclave_test_max_count # same as max test since min already satisfied
        local_test = LOCAL.local_test_min_max_count
    else:
        raise Exception, "inconsistent set of threshold parameters passed"
    return build_region, enclave_test, local_test




