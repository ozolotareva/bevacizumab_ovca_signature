import pandas as pd
import numpy as np
import os,sys

from bic_survival_tests import test_samples_predictive

def do_permutations_for_predictive(surv_data,
                                   samples,
                                   covariates = [],
                                   treatment_col = "bevacizumab",
                                   time_col = "_time",
                                   event_col = "_event",
                                   n_perm = 1000,
                                   seed = 42
                                  ):
    perm_results = []
    np.random.seed(seed)
    for i in range(n_perm):
        perm_seed = np.random.randint(0,1000000)
        # make sample annotation table with permuted event column
        surv_perm = surv_data.copy()
        np.random.seed(perm_seed)
        
        # permute events *within* treatment groups
        for event in set(surv_perm[treatment_col].values):
            surv_perm.loc[surv_perm[treatment_col]==event,event_col] = np.random.permutation(surv_perm.loc[surv_perm[treatment_col]==event,event_col].values)

        hr, ci, pval, covar_excluded = test_samples_predictive(surv_perm,
                                    samples, # compare treatments in this sample group
                                    covariates = [],
                                    treatment_col = "bevacizumab",
                                    time_col = time_col,
                                    event_col = event_col
                                   )
        perm_results.append({"seed":perm_seed,
                             "HR": hr,
                             "CI":ci,
                             "pval":pval,
                             "covar_excluded": covar_excluded
                            })
    perm_results = pd.DataFrame.from_records(perm_results).sort_values(by="HR")
    return perm_results

def calc_perm_pval(perm_results,hr):
    if hr <1:
        n_rand = perm_results.loc[perm_results["HR"]<hr,:].shape[0]
    elif hr >1:
        n_rand = perm_results.loc[perm_results["HR"]>hr,:].shape[0]
    else:
        return 1
    perm_pval = (n_rand+1)/(perm_results.shape[0])
    rand_avg_hr = perm_results["HR"].mean()
    return perm_pval,rand_avg_hr