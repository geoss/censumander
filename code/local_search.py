'''
Various pieces needed to conduct the local search (aka "swapping") stage.
'''

__author__ = "David C. Folch <dfolch@gmail.com>, Seth E. Spielman <seth.spielman@colorado.edu>"


import copy
from operator import itemgetter
import numpy as np
from utils import sum_squares, list_copy
import pysal


class Mover:
    '''
    Fancy class to organize the components of a candidate move or the tabu list. 
    '''
    def __init__(self, aid, rid_orig, rid_neig, ssds, id2region, short, tabu_lock):
        self.aid = aid
        self.rid_orig = rid_orig
        self.rid_neig = rid_neig
        self.ssd_orig = ssds[rid_orig]
        self.ssd_neig = ssds[rid_neig]
        # we don't really need to copy the whole dictionary; we just need to
        #     copy the values, and then change one element in that list; 
        #     probably a clever way to do this to save time (and possibly RAM)
        id2region_local = id2region.copy()
        if tabu_lock:
            # when locking in a move, we don't allow aid to return to rid_orig
            self.key_intermediate = (aid, rid_orig)
            # when locking in a move, we don't allow aid to return to an earlier map
            id2region_local[aid] = rid_orig
            self.key_short = tuple(id2region_local.values())
        else:
            # when testing a move, we check if aid wants to return to a tabu rid
            self.key_intermediate = (aid, rid_neig)
            # when testing a move, we check if aid wants to return to an earlier map
            id2region_local[aid] = rid_neig
            self.key_short = tuple(id2region_local.values())
        self.set_key(short)
    # stuff to allow flipping between short and intermediate tabu lists
    def key_short_func(self):
        '''
        This hash is for search intensification since each instance of this
        class eliminates one possible solution. This class takes a snapshot of
        the whole map, not the move. This means that this move can be
        repeated, so long as we don't return to this exact map.

        assuming id2region_local.values() is always in the same order since built with id2region.copy(); based on these links
        http://stackoverflow.com/questions/2053021/is-the-order-of-a-python-dictionary-guaranteed-over-iterations
        http://stackoverflow.com/questions/3666237/are-order-of-keys-and-values-in-python-dictionary-guaranteed-to-be-the-same?rq=1
        '''
        return self.key_short
    def key_intermediate_func(self):
        '''
        This hash is for search diversification since each instance of this
        class eliminates more than one possible solution. This class takes a
        snapshot of the area and its former region, not the whole map. This
        means that any move trying to put this area back into this region is
        disallowed; this implies that not only this map cannot be repeated,
        but all maps with this area in this region cannot be repeated.
        '''
        return self.key_intermediate
    def set_key(self, value):
        self.short = value
        if value:
            self.__key = self.key_short_func
        else:
            self.__key = self.key_intermediate_func
    # some extra stuff to use instances of this class in sets
    def __eq__(x, y):
        return x.__key() == y.__key()
    def __hash__(self):
        return hash(self.__key())


def local_search(regions, id2region, w, count_th_min, count_th_max, count_est, target_th,\
                          target_parts, target_est, exclude, get_cv,\
                          local_test, local_params, cv_exclude):
    '''
    Starting with an base solution, swap areas in an attempt to reduce the
    global sum of squares, while maintaining threshold constraints.

    At each step it aways takes the best possible move, meaning that it is
    entirely deterministic given a fixed starting point and fixed tabu search
    parameters.
    '''
    # clean up parameters
    if not local_params:
        local_params = {}
    else:
        local_params = copy.deepcopy(local_params)    
    ####################################################
    # default settings based on Folch and Spielman (IJGIS, 2014)
    if 'max_swaps' not in local_params:
        local_params['max_swaps'] = w.n
    if 'max_chain' not in local_params:
        local_params['max_chain'] = w.n * 0.50
    if 'max_tabu' not in local_params:
        local_params['max_tabu'] = w.n * 0.40
    if 'change_thresh' not in local_params:
        local_params['change_thresh'] = 0.0000001
    ####################################################
    # baseline values
    solution = copy.deepcopy(regions)
    ssds = [sum_squares(region, target_est) for region in regions]
    memory_global_ssds = [sum(ssds)]
    ssd_global = sum(ssds)
    ssd_base_solution = sum(ssds)
    ssd_start = sum(ssds)
    ssd_last10 = [ssd_start*10] * 9
    last10_feasible = True
    memory_solutions = [list_copy(solution)]
    memory_id2region = [id2region.copy()]
    memory_ssds = [copy.copy(ssds)]
    # SSD threshold test: some SSD values get VERY big and cause rounding
    # problems when comparing two values; this hack sets the magnitude for
    # differentiating two values at some very small negative value relative to
    # the magnitude of the SSD values
    ssd_test = len(str(int(ssd_global))) # find number of digits in front of decimal
    if ssd_test > 12:
        # if the SSDs are very big, essentially treat them as integers
        ssd_test = -1
    else:
        # if the SSDs are in some lower range then create a very small float
        ssd_test = -float('1e-'+str(12 - ssd_test))
    ##### setup parameters #########################
    max_swaps = local_params['max_swaps']
    max_chain = local_params['max_chain']
    max_tabu = local_params['max_tabu']
    last10_threshold = local_params['change_thresh']
    ################################################
    tabu = []
    memory_tabu = []
    total_swaps = 1
    keep_going = True  # serves as an override to stop the algorithm in unusual cases
    exit = 'not recorded'
    short = False  # if True then run intense search
    intense_start_state = 'did not shift to intense tabu'
    # note: for these purposes a swap is a single improvement in SSD even if
    #       it takes a long chain to get there
    # note: the chain restarts each time a feasible SSD improving swap is found
    # note: we set a maximum number of feasible swaps so that the script
    #       doesn't run forever



    while True:
        ################################
        # conditions for ending or switching algorithm
        ################################
        if not keep_going:
            if short:
                # this block catches the less usual reasons for stopping the
                # algorithm; keep_going was triggered at some point in the
                # last pass through, and since it was in the intense search
                # stage it is time to stop the algorithm;
                # rollback to last good result and exit
                solution = memory_solutions[0]
                id2region = memory_id2region[0]
                break
        if total_swaps > max_swaps:
            # if it ever gets to max_swaps (no matter if it's in the diverse
            # or intense search stage), then it's time to stop;
            # rollback to last good result and exit
            solution = memory_solutions[0]
            id2region = memory_id2region[0]
            exit = 'max swaps'
            break
        if len(memory_solutions) > max_chain or not keep_going:
            if not short:
                # finished diverse search (aka "intermediate"), rollback to 
                # last good result and begin intense (aka "short") search
                ##print '\n\n*****************************************************'
                ##print "switching to intense tabu search", total_swaps, len(memory_solutions), sum(memory_ssds[0])
                ##print '*****************************************************\n\n'
                intense_start_state = total_swaps, len(memory_solutions), sum(memory_ssds[0])
                tabu = memory_tabu[:]
                for i in tabu:
                    i.set_key(True)
                short = True
                keep_going = True
                # reset baseline values
                solution = memory_solutions[0]
                memory_solutions = [list_copy(solution)]
                id2region = memory_id2region[0]
                memory_id2region = [id2region.copy()]
                ssds = memory_ssds[0]
                memory_ssds = [ssds[:]]
                memory_global_ssds = [sum(ssds)]
                ssd_global = sum(ssds)
                ssd_base_solution = sum(ssds)
                ssd_start = sum(ssds)
            else:
                # finished intense search, rollback to last good result and exit
                solution = memory_solutions[0]
                id2region = memory_id2region[0]
                exit = 'max chain'
                break



        ################################
        # main section of algorithm
        ################################

        ##print 'total swaps', total_swaps 
        ##print 'chain length', len(memory_solutions) 
        ##print 'tabu length', len(tabu)
        ##print '\tstart SSD', memory_global_ssds[-1]
        ##if total_swaps%100==0: print 'swaps', total_swaps 

        # build list of candidate moves
        candidates = []
        for rid, region in enumerate(solution):
            if len(region) > 1:
                # can only move an area out if there are multiple areas in region
                for i in region:
                    neighbors = w.neighbors[i]
                    # eliminate neighbors in i's region
                    outside = set(neighbors).difference(region)
                    # eliminate neighbors in excluded list
                    outside = outside.difference(exclude)
                    # eliminate redundant neighboring regions
                    unique_regions = {id2region[j] for j in outside}  # returns a set
                    # add area moves to candidate list
                    for rid_neig in unique_regions:
                        candidates.append(Mover(i, rid, rid_neig, ssds, id2region, short, False))
       
        ##print '\tnumber of candidates', len(candidates)
        # candidates represents all possible single moves
        # note: a particular area may be in multiple candidates
        if not candidates:
            # stop swapping if no more candidates can be found
            # roll back solution and id2regions to best feasible solution
            # note: I suspect this little block of code has no impact on the
            #       results: if there are no candidates the else part of the
            #       while-else loop that follows with catch it. However, I'm 
            #       leaving it in case it catches some case I cannot remember.
            solution = list_copy(memory_solutions[0])
            id2region = memory_id2region[0].copy()
            keep_going = False
            exit = 'no candidates'
       
        # measure SSD for each candidate move
        test = []
        for candidate in candidates:
            region_orig = solution[candidate.rid_orig][:]
            region_neig = solution[candidate.rid_neig][:]
            region_orig.remove(candidate.aid)
            region_neig.append(candidate.aid)
            ssd_orig_r = sum_squares(region_orig, target_est)
            ssd_neig_r = sum_squares(region_neig, target_est)
            candidate.ssd_orig_r = ssd_orig_r
            candidate.ssd_neig_r = ssd_neig_r
            test.append(ssd_start - (candidate.ssd_orig + candidate.ssd_neig)\
                                   + (ssd_orig_r + ssd_neig_r))
            ##print '\t\t\tCandidate, CV', candidate.aid, test[-1]
        
        # identify which feasible candidate move results in lowest global SSD
        test_zip = zip(test, candidates)
        # sort largest to smallest, then pop from the end
        test_zip = sorted(test_zip, key=itemgetter(0), reverse=True)
        candidates = [i[1] for i in test_zip]
        test_sorted = [i[0] for i in test_zip]
        update_tabu = False
        tabu_lock = False
        last10_feasible = True

        while candidates:
            best_test = test_sorted.pop()
            best_cand = candidates.pop()
            ##print '\t\t', best_cand.aid, best_cand.rid_orig, best_cand.rid_neig 
            ##print '\t\ttesting SSD', total_swaps, len(memory_solutions), best_test, ssd_global, best_test -ssd_global
            solution = list_copy(memory_solutions[-1])
            solution[best_cand.rid_orig].remove(best_cand.aid)
            solution[best_cand.rid_neig].append(best_cand.aid)
            # check candidate feasibility here since the tests are expensive
            # only consider moves that meet feasibility thresholds
            candidate_feasible = False
            if local_test(solution, target_parts, target_th,\
                          count_est, count_th_min, count_th_max, get_cv,\
                          memory_solutions[-1], cv_exclude):
                # check if proposed move will split its region
                if pysal.region.check_contiguity(w,\
                             memory_solutions[-1][best_cand.rid_orig],\
                             best_cand.aid):
                    candidate_feasible = True
            # only proceed if the candidate meets all requirements
            if candidate_feasible:
                ##print '\t\t', best_cand.aid, best_cand.rid_orig, best_cand.rid_neig 
                ##print '\t\ttesting SSD', best_test, ssd_global, best_test -ssd_global
                ##print '\t\tSSD components', ssd_start, "- (",\
                ##           best_cand.ssd_orig, "+", best_cand.ssd_neig, ") + (",\
                ##           best_cand.ssd_orig_r, "+", best_cand.ssd_neig_r, ")"
                ##if best_test - ssd_global < 0 and best_test - ssd_global > ssd_test:
                ##    print total_swaps, best_test, ssd_global, best_test - ssd_global, ssd_test
                if best_test - ssd_global < ssd_test:
                    # candidate improves global SSD: reset memory
                    # note: we ignore tabu list for feasible candidates that
                    #       improve SSD
                    ##print '\t\t>>> STOP: Improved global SSD'
                    ##print '\t\t', best_cand.aid, best_cand.rid_orig, best_cand.rid_neig 
                    ##print '\t\tsuccessful swap chain length', len(memory_solutions)
                    # update tabu list
                    update_tabu = True
                    tabu_lock = True
                    # reset solution chain to best move
                    memory_solutions = [list_copy(solution)]
                    # reset id2region chain to best move
                    id2region[best_cand.aid] = best_cand.rid_neig
                    memory_id2region = [id2region.copy()]
                    # reset SSDs chain to best move
                    ssds[best_cand.rid_orig] = best_cand.ssd_orig_r
                    ssds[best_cand.rid_neig] = best_cand.ssd_neig_r
                    memory_ssds = [ssds[:]]
                    memory_global_ssds = [best_test]
                    ssd_global = best_test
                    ssd_start = best_test
                    # test total change
                    ssd_last = ssd_last10.pop(0)
                    ssd_last10.append(ssd_global)
                    ##print '\t\tSSD change, over last 10:', (ssd_last - ssd_global)/ssd_last
                    if (ssd_last - ssd_global)/ssd_last < last10_threshold:
                        last10_feasible = False
                    total_swaps += 1
                    break
                elif best_cand in tabu:
                    # skip this candidate: it doesn't improve SSD and is in tabu list
                    ##print '\t\t>>> SKIP: In tabu list, and does not improve SSD'
                    pass
                else:
                    # candidate does not improve global SSD and is not in tabu list: add to memory
                    # add solution to chain of sub-optimal solutions in the hope of
                    # eventually finding an optimal solution later
                    ##print '\t\t>>> ADD: Does not improve global SSD'
                    ##print '\t\t', best_cand.aid, best_cand.rid_orig, best_cand.rid_neig 
                    update_tabu = True
                    # update solution based on best move
                    memory_solutions.append(list_copy(solution))
                    # update id2region dictionary based on best move
                    id2region[best_cand.aid] = best_cand.rid_neig
                    memory_id2region.append(id2region.copy())
                    # update SSDs based on best move
                    ssds[best_cand.rid_orig] = best_cand.ssd_orig_r
                    ssds[best_cand.rid_neig] = best_cand.ssd_neig_r
                    memory_ssds.append(ssds[:])
                    memory_global_ssds.append(best_test)
                    ssd_start = best_test
                    break
            else:
                # skip this candidate: it is not feasible
                ##print '\t\t>>> SKIP: Not feasible'
                #solution = list_copy(memory_solutions[-1])
                pass
        else:
            # stop swapping if no viable candidate can be found
            # roll back solution and id2regions to best feasible solution
            solution = list_copy(memory_solutions[0])
            id2region = memory_id2region[0].copy()
            keep_going = False
            exit = 'no viable candidates'
        if not last10_feasible:
            # the current improvement in SSD is less than __% since 10 moves ago
            ##print '\tlow change in SSD'
            keep_going = False
            exit = 'below threshold'
        if update_tabu:
            # add map to tabu list
            tabu.append(Mover(best_cand.aid, best_cand.rid_orig,\
                              best_cand.rid_neig, memory_ssds[-1][:],\
                              id2region, short, True))
            ##print '\t\t\tlast tabu', tabu[-1].aid, tabu[-1].rid_orig, tabu[-1].rid_neig
            if len(tabu) > max_tabu:
                tabu.pop(0)
            if tabu_lock:
                # tabu list associated with last improving move
                memory_tabu = tabu[:]
    ##print intense_start_state, (total_swaps, len(memory_solutions), sum(memory_ssds[0]))
    return solution, id2region, exit



###################################################################
# Four different threshold criteria for defining feasible regions #
###################################################################
def local_test_cv_only(solution, target_parts, target_th,\
                         count_est, count_th_min, count_th_max, get_cv, last_solution, cv_exclude):
    '''
    Need to be below CV threshold.
    '''
    changed_regions = []
    for rid in xrange(len(solution)):
        if solution[rid] != last_solution[rid]:
            changed_regions.append(rid)
    for rid in changed_regions:
        region = np.array(solution[rid], copy=False, dtype=np.int32)
        if np.any(get_cv(region, target_parts, cv_exclude) > target_th):
            ##print '\t\t\tFailed CV', get_cv(region, target_parts, cv_exclude)
            return False
    return True

def local_test_count_only(solution, target_parts, target_th,\
                         count_est, count_th_min, count_th_max, get_cv, last_solution, cv_exclude):
    '''
    Need to be above count threshold.
    '''
    for region in solution:
        if count_est[region].sum() < count_th_min:
            ##print '\t\t\tFailed Count', count_est[region].sum()
            return False
    return True

def local_test_min_count(solution, target_parts, target_th,\
                         count_est, count_th_min, count_th_max, get_cv, last_solution, cv_exclude):
    '''
    Need to be above count threshold, and below CV threshold.
    '''
    changed_regions = []
    for rid in xrange(len(solution)):
        if solution[rid] != last_solution[rid]:
            changed_regions.append(rid)
    for rid in changed_regions:
        region = np.array(solution[rid], copy=False, dtype=np.int32)
        if count_est[region].sum() < count_th_min:
            ##print '\t\t\tFailed Count', count_est[region].sum()
            return False
        if np.any(get_cv(region, target_parts, cv_exclude) > target_th):
            ##print '\t\t\tFailed CV', get_cv(region, target_parts)
            return False
    return True

def local_test_max_count(solution, target_parts, target_th,\
                         count_est, count_th_min, count_th_max, get_cv, last_solution, cv_exclude):
    '''
    Need to be below count threshold, and below CV threshold.
    '''
    changed_regions = []
    for rid in xrange(len(solution)):
        if solution[rid] != last_solution[rid]:
            changed_regions.append(rid)
    for rid in changed_regions:
        region = np.array(solution[rid], copy=False, dtype=np.int32)
        # test all region sizes first since get_cv() is much slower
        if count_est[region].sum() > count_th_max:
            ##print '\t\t\tFailed Count', count_est[region].sum()
            return False
        if np.any(get_cv(region, target_parts, cv_exclude) > target_th):
            ##print '\t\t\tFailed CV', get_cv(region, target_parts)
            return False
    return True

def local_test_min_max_count(solution, target_parts, target_th,\
                             count_est, count_th_min, count_th_max, get_cv, last_solution, cv_exclude):
    '''
    Need to get above min count threshold and stay below max count threshold,
    and below CV threshold.
    '''
    changed_regions = []
    for rid in xrange(len(solution)):
        if solution[rid] != last_solution[rid]:
            changed_regions.append(rid)
    for rid in changed_regions:
        region = np.array(solution[rid], copy=False, dtype=np.int32)
        count = count_est[region].sum()
        if count < count_th_min:
            ##print '\t\t\tFailed Min Count', count
            return False
        if count > count_th_max:
            ##print '\t\t\tFailed Max Count', count
            return False
        if np.any(get_cv(region, target_parts, cv_exclude) > target_th):
            ##print '\t\t\tFailed CV', get_cv(region, target_parts, cv_exclude)
            return False
    return True




