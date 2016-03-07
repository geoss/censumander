'''
Various pieces needed for the base region construction stage.
'''

__author__ = "David C. Folch <dfolch@gmail.com>, Seth E. Spielman <seth.spielman@colorado.edu>"


import copy
import numpy as np
import utils as UTILS


def cardinality_sorter(ids, w):
    # sort on cardinality first, then distance from anchor
    ##### probably room to optimize this ######
    ids_card = [(w.cardinalities[i],index,i) for index, i in enumerate(ids)]
    ids_card.sort()  
    ids = [i[2] for i in ids_card]
    return ids



def base_region_iterator(w, count_th_min, count_th_max, count_est, target_th, target_est,\
                         exclude, auto_exclude, get_cv, base_solutions,\
                         target_parts, build_region, enclave_test ,kd, points,
                         anchor, cardinality, cv_exclude):
    '''
    Build a fixed number of solutions, and pick the solution with the
    maximum number of regions.  If there is a tie for the most number of
    regions, then pick the solution with the lowest sum of squares.
    '''
    ##################################
    # setup the ID order for each base solution
    ##################################
    ids_prepped = []
    ids = range(w.n)
    # get the excluded IDs out of the main ID pool
    if exclude:
        for i in exclude:
            ids.remove(i)        
    else:
        exclude = []
    if kd:
        # run the Vickrey (1961) approach that builds regions from the outside in
        if type(anchor)==int:  # only works on integers now
            # user passes in a unique anchor
            ids = [anchor]
            base_solutions = 1
        elif anchor is None:
            # create a randomly ordered list of anchor IDs
            ids = np.random.permutation(ids).tolist()
            if len(ids) < base_solutions:
                base_solutions = len(ids)
                print "WARNING: number of included areas (%c) is less than base_solutions (%c); proceeding with %c base solutions"%(len(ids), base_solutions, len(ids))
        else:
            raise Exception, 'invalid value passed to anchor parameter (must be int or None)'
        # create a list of all n IDs for each base_solution to be run 
        for anchor_iter in ids[0:base_solutions]:
            kd_ordering = kd.query(points[anchor_iter], k=w.n, p=2)
            ids_kd = kd_ordering[1].tolist()  
            ids_kd.reverse()  # order seeds so that furthest from the anchor is first, the anchor is last
            if cardinality:
                ids_kd = cardinality_sorter(ids_kd, w)
            ids_prepped.append(ids_kd)
    else:
        # build regions randomly
        for run in range(base_solutions):
            ids = np.random.permutation(ids).tolist()
            if cardinality:
                ids = cardinality_sorter(ids, w)
            ids_prepped.append(ids)




    ##################################
    # build the base solutions and select the best    
    ##################################
    feasible = False
    best_num_regions = 0
    for ids in ids_prepped:
        regions, id2region, ids, enclaves = base_construction(\
                                          w, count_th_min, count_th_max, count_est, target_th,\
                                          exclude, auto_exclude, get_cv, ids,\
                                          target_parts, build_region, enclave_test,\
                                          cardinality, cv_exclude)
        if id2region:
            # only test feasible solutions
            feasible = True
            test_num_regions = len(regions)
            ##print '\tnum regions', test_num_regions
            if  test_num_regions > best_num_regions:
                # always take the solution with higher number of regions
                ##print '\t\t**** more regions', test_num_regions
                best_num_regions = test_num_regions
                best_regions = regions
                best_id2region = id2region
                best_ids = ids
                best_enclaves = enclaves
                best_ssd = 0
                if target_est is not None:
                    for i in regions:
                        best_ssd += UTILS.sum_squares(i, target_est)
            elif test_num_regions == best_num_regions and target_est is not None:
                # if same number of regions, take the solution with lowest
                # SSD; if there are no target_est variables then there is no
                # way to break ties (we just keep the first one)
                ##print '\t\t**** same regions', test_num_regions
                test_ssd = 0
                for i in regions:
                    test_ssd += UTILS.sum_squares(i, target_est)
                if test_ssd < best_ssd:
                    ##print '\t\t\t**** improved SSD', test_ssd
                    best_regions = regions
                    best_id2region = id2region
                    best_ids = ids
                    best_enclaves = enclaves
                    best_ssd = test_ssd
                else:
                    ##print '\t\t\t**** same or worse SSD'
                    pass
        else:
            ##print "\tno solution"
            pass



    ##################################
    # return final result
    ##################################
    if feasible:
        # got a feasible solution
        ##print "total regions", len(best_regions)
        if auto_exclude:
            # if the best solution has remaining enclaves, add them to the exclude list
            exclude = copy.copy(exclude)
            exclude.extend(best_ids)
        return best_regions, best_id2region, exclude, best_enclaves
    else:
        # no feasible solution; return the problem IDs by adding to the exclude list 
        print "no feasible solution"
        exclude = copy.copy(exclude)
        exclude.extend(ids)
        return [], False, exclude, enclaves
    

def base_construction(w, count_th_min, count_th_max, count_est, target_th,\
                      exclude, auto_exclude, get_cv, ids,\
                      target_parts, build_region, enclave_test,\
                      cardinality, cv_exclude):
    '''
    Construct a full initial solution. This function essentially has two
    parts. The first chooses seeds one at a time and builds regions
    around the seed. The second part attaches remaining areas the cannot form
    regions (aka "enclaves") to adjacent existing regions. This is intended to
    run fast so there is no optimization in the region construction.  This 
    function is run many times and the best result forms the basis of a later 
    round of swapping.  
    '''
    # set some global variables
    regions = []
    id2region = {}
    used_ids = set(exclude)  # keeps the excluded IDs out of the neighbors lists
    # set while loop variables
    loop_feasible = True
    loop_count = 0           # just for printing purposes
    while loop_feasible==True:
        # keeping looping until no more areas can be assigned
        ##print '\tregions loop', loop_count
        ##print '\tnumber of IDs remaining', len(ids)
        loop_count += 1
        loop_feasible = False
        remove_ids = []       # these will be popped off at the end of the loop
        for seed in ids:
            # iterate over available areas
            ##print '\t\tseed', seed
            if seed in used_ids:
                # don't build a new region around an area already in a region
                continue
            # build one region at a time
            feasible, region = build_region(w, count_th_min, count_th_max, count_est,\
                                            target_th, get_cv, target_parts,\
                                            seed, used_ids, cv_exclude)
            if feasible:
                # region meets both the count and CV thresholds
                ##print '\t\t\tfeasible'
                # cleanup and prep for next seed
                loop_feasible = True
                used_ids = used_ids.union(region)
                regions.append(region)
                region_id = len(regions) - 1
                remove_ids.extend(region)
                for i in region:
                    id2region[i] = region_id
        ids_remove = ids.remove   # localize the fuction
        for i in remove_ids:
            # once an entire loop is run, then remove IDs from the master list
            ##print '\\ttremove', i
            #ids.remove(i)
            ids_remove(i)

    # start procedure to match enclaves (a.k.a. leftovers) to existing regions
    loop_feasible = True
    if w.n == len(ids):
        # if no regions were created, then there is nothing to join enclaves to
        loop_feasible = False
    loop_count = 0             # just for printing
    ##print '\tnumber of regions', len(regions)
    ##print '\tnumber of remaining enclave IDs', len(ids)
    enclaves = len(ids)
    while loop_feasible==True:
        # keep looping over leftover IDs until none can be matched to an existing region
        ##print '\tleftover loop', loop_count
        ##print '\t\tnumber of remaining enclave IDs', len(ids)
        loop_count += 1
        loop_feasible = False
        remove_ids = []
        for leftover in ids:
            ##print '\tleftover', leftover
            neighbors = w.neighbors[leftover]
            neighbors = np.random.permutation(neighbors)
            '''
            this only allows an area to join a region if the revised region meets
            the CV and count thresholds; this means that some areas might not be 
            able to join a region, and the run will fail
            '''
            for link in neighbors:
                # iterate over neighbors of leftover to find a feasible region match
                ##print '\t\t\tlink', link
                if link in id2region.keys():
                    # the link must already be in a region
                    ##print '\t\t\tlink available'
                    link_region_id = id2region[link]
                    link_region = regions[link_region_id][:]
                    ##print '\t\t\t\tstart CV', get_cv(link_region, target_parts, cv_exclude)
                    link_region.append(leftover)
                    # test if the modified region still meets CV and count thresholds
                    if enclave_test(link_region, target_parts, target_th,\
                                    count_est, count_th_max, get_cv, cv_exclude):
                        # cleanup if the leftover is successfully matched
                        ##print '\t\t\t\tend CV', get_cv(link_region, target_parts, cv_exclude)
                        ##print '\t\t\t\tlink feasible'
                        regions[link_region_id] = link_region
                        used_ids.add(leftover)
                        remove_ids.append(leftover)
                        id2region[leftover] = link_region_id
                        loop_feasible = True
                        break
                    ##else:
                        ##print '\t\t\t\tend CV', get_cv(link_region, target_parts, cv_exclude)
                        ##print '\t\t\t\tlink not feasible'
        for i in remove_ids:
            # once an entire loop is run, then remove IDs from the master list
            ids.remove(i)
    if ids:
        if len(ids) / (w.n*1.0) < auto_exclude:
            ##print '\tFEASIBLE with auto exclude threshold'
            ##print '\t\tremaining IDs', ids
            ##print '\t\tnumber of remaining IDs', len(ids), '; number of regions', len(regions)
            return regions, id2region, ids, enclaves
        else:
            ##print '\tNOT FEASIBLE'
            ##print '\t\tremaining IDs', ids
            ##print '\t\tnumber of remaining IDs', len(ids), '; number of regions', len(regions)
            return False, False, ids, enclaves
    else:
        ##print '\tFEASIBLE'
        ##print '\t\tnumber of regions', len(regions)
        return regions, id2region, ids, enclaves


######################################################################
# Different ways to build a single region based on the four possible #
# combinations of threshold criteria for defining feasible regions   #
######################################################################
def build_region_cv_only(w, count_th_min, count_th_max, count_est, target_th,\
                          get_cv, target_parts, seed, used_ids, cv_exclude):
    '''
    Need to get below CV threshold.
    '''
    # set up the parts to build a new region
    region = [seed]
    neighbors = get_neighbors(w, seed, [], region, used_ids)
    return region_cv_only(w, target_th, get_cv, target_parts, seed, used_ids,\
                          neighbors, region, cv_exclude)

def build_region_count_only(w, count_th_min, count_th_max, count_est, target_th,\
                             get_cv, target_parts, seed, used_ids, cv_exclude):
    '''
    Need to get above count threshold.
    '''
    # set up the parts to build a new region
    region = [seed]
    neighbors = get_neighbors(w, seed, [], region, used_ids)
    return region_min_count(w, count_th_min, count_est, seed, used_ids,\
                            neighbors, region)

def build_region_min_count(w, count_th_min, count_th_max, count_est, target_th,\
                          get_cv, target_parts, seed, used_ids, cv_exclude):
    '''
    Need to get above count threshold, and below CV threshold.
    '''
    # set up the parts to build a new region
    region = [seed]
    neighbors = get_neighbors(w, seed, [], region, used_ids)
    # build up region that meets count threshold
    feasible, region = region_min_count(w, count_th_min, count_est,\
                                        seed, used_ids, neighbors, region)
    if feasible:
        # continue building region to meet CV threshold
        feasible, region = region_cv_only(w, target_th, get_cv,\
                                          target_parts, seed, used_ids,\
                                          neighbors, region, cv_exclude)
    return feasible, region

def build_region_max_count(w, count_th_min, count_th_max, count_est, target_th,\
                          get_cv, target_parts, seed, used_ids, cv_exclude):
    '''
    Need to stay below count threshold, and below CV threshold.
    '''
    # set up the parts to build a new region
    region = [seed]
    neighbors = get_neighbors(w, seed, [], region, used_ids)
    feasible, region = region_cv_only(w, target_th, get_cv,\
                                      target_parts, seed, used_ids,\
                                      neighbors, region, cv_exclude)
    if feasible:
        if count_est[region].sum() > count_th_max:
            feasible = False
    return feasible, region



def build_region_min_max_count(w, count_th_min, count_th_max, count_est, target_th,\
                              get_cv, target_parts, seed, used_ids, cv_exclude):
    '''
    Need to get above min count threshold and stay below max count threshold,
    and below CV threshold.
    '''
    # set up the parts to build a new region
    region = [seed]
    neighbors = get_neighbors(w, seed, [], region, used_ids)
    # build up region that meets count threshold
    feasible, region = region_min_count(w, count_th_min, count_est,\
                                        seed, used_ids, neighbors, region)
    if feasible:
        # check to ensure region is below max count threshold
        if count_est[region].sum() > count_th_max:
            ##print '\t\t\tnot feasible, max1,', len(region), count_est[region].sum()
            ##print '\t\t\t', region
            feasible = False
        if feasible:
            # continue building region to meet CV threshold
            feasible, region = region_cv_only(w, target_th, get_cv,\
                                              target_parts, seed, used_ids,\
                                              neighbors, region, cv_exclude)
            if feasible:
                # check to ensure region is below max count threshold
                if count_est[region].sum() > count_th_max:
                    ##print '\t\t\tnot feasible, max2,', len(region), count_est[region].sum()
                    ##print '\t\t\t', region
                    feasible = False
            else:
                ##print '\t\t\tnot feasible, CV,', len(region), count_est[region].sum()
                ##print '\t\t\t', region
                pass
    else:
        ##print '\t\t\tnot feasible, min,', len(region), count_est[region].sum()
        ##print '\t\t\t', region
        pass
    return feasible, region




def region_cv_only(w, target_th, get_cv, target_parts, seed, used_ids,\
                   neighbors, region, cv_exclude):
    '''
    Construct a region based on a CV constraint only
    '''
    feasible = True
    cv = get_cv(region, target_parts, cv_exclude)
    while np.any(cv > target_th):
        # keep adding areas until region meets CV threshold
        ##print '\tregion:', region
        ##print '\tneighbors:', neighbors
        if not neighbors:
            # no more neighbors available, so this region is infeasible 
            ##print '\t\tnot feasible, CV; region size', len(region) 
            feasible = False
            break
        # grab the first neighbor in the list, and add to region
        # NOTE: to promote speed, we do not hunt for the "best" neighbor
        add_on = neighbors.pop(0)
        region.append(add_on)
        ##print '\t\t\tadd_on', add_on
        cv = get_cv(region, target_parts, cv_exclude)
        ##print '\t\t', add_on, cv
        # add new potential neighbors as a result of add_on to bottom of list
        neighbors = get_neighbors(w, add_on, neighbors, region, used_ids)
    ##print '\t\tend CV', cv
    ##print '\t\t\tfeasible', feasible
    return feasible, region

def region_min_count(w, count_th_min, count_est, seed, used_ids, neighbors, region):
    '''
    Construct a region based on a count constraint only
    '''
    feasible = True
    count = copy.copy(count_est[seed])
    while count < count_th_min:
        # keep adding areas until region meets count threshold
        if not neighbors:
            # no more neighbors available, so this region is infeasible 
            ##print '\t\tnot feasible, count <<<<<<<<<<<<<<<' 
            feasible = False
            break
        # grab the first neighbor in the list, and add to region
        # NOTE: pick from the top to promote compactness of regions
        add_on = neighbors.pop(0)
        region.append(add_on)
        ##print '\t\tadd_on', add_on
        count += count_est[add_on]
        # add new potential neighbors as a result of add_on to bottom of list
        neighbors = get_neighbors(w, add_on, neighbors, region, used_ids)
    ##print '\t\t\tfeasible', feasible
    return feasible, region
########################



###################################################################
# Three different tests for feasible enclave assignment to region #
###################################################################
def enclave_test_cv_only(region, target_parts, target_th,\
                         count_est, count_th_max, get_cv, cv_exclude):
    if np.any(get_cv(region, target_parts, cv_exclude) > target_th):
        ##print '\t\t\tFailed CV', get_cv(region, target_parts, cv_exclude)
        return False
    return True

def enclave_test_count_only(region, target_parts, target_th,\
                            count_est, count_th_max, get_cv, cv_exclude):
    return True

def enclave_test_max_count(region, target_parts, target_th,\
                           count_est, count_th_max, get_cv, cv_exclude):
    if np.any(get_cv(region, target_parts, cv_exclude) > target_th):
        ##print '\t\t\tFailed CV', get_cv(region, target_parts, cv_exclude)
        return False
    if count_est[region].sum() > count_th_max:
        ##print '\t\t\tFailed Count', count_est[region].sum()
        return False
    return True
########################




def get_neighbors(w, seed, neighbors, region, used_ids): 
    '''
    update the current neighbors list by appending a randomized list
    containing the neighbors of the newest addition to the region, but that
    excludes any ids that have already been used; we add the new neighbors to the
    end to promote compact regions
    '''
    new_neighbors = set(w.neighbors[seed])
    new_neighbors = new_neighbors.difference(region)
    new_neighbors = new_neighbors.difference(used_ids)
    new_neighbors = new_neighbors.difference(neighbors)
    new_neighbors = np.random.permutation(list(new_neighbors)).tolist()
    neighbors.extend(new_neighbors)
    return neighbors



