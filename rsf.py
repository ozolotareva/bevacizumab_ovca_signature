from sklearn import set_config
from sksurv.datasets import  get_x_y
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

set_config(display="text")

def make_binary_bics(bics, all_sample_set):
    # 1- high expression, 0 -low
    bin_bics = np.zeros((bics.shape[0],len(all_sample_set)))
    bin_bics = pd.DataFrame(bin_bics, 
                            index=["bic"+str(x) for x in bics.index],
                            columns=sorted(all_sample_set))
    for bic_id in bics.index.values:
        s = sorted(bics.loc[bic_id,"samples"].intersection(all_sample_set))
        if bics.loc[bic_id,"direction"] == "DOWN":
            bin_bics.loc["bic"+str(bic_id),s] = -1
            bin_bics.loc["bic"+str(bic_id),:] += 1
        else:
            bin_bics.loc["bic"+str(bic_id),s] = 1
    return bin_bics.T


def prepare_x_y_for_RSF(bics, annot,
                        min_n_samples = 10, 
                        treatment_col = "bevacizumab",
                        treatment_value =1,
                        covariates = [],
                        surv_cols = ["_event","_time"]):
    """For each bicluster, binary variable is added to X, where
    1- High expression of bicluster genes, 0 - low.
    """
    cols = covariates+surv_cols
    # keep only samples under treatment
    annot_treated = annot.copy().loc[annot[treatment_col]==treatment_value,cols]
    # make binary biclusters
    bics_binary = make_binary_bics(bics.copy(), set(annot_treated.index.values))
    # remove biclusters with <`min_n_samples` samples in any group (bic,bg)xtreatment
    counts = bics_binary.sum()
    counts = counts[min_n_samples<=counts]
    counts = counts[counts<=bics_binary.shape[0]-min_n_samples]
    passed_bics = counts.index.values
    not_passed_bics = set(bics_binary.columns.values).difference(set(passed_bics))
    if len(not_passed_bics)>0:
        bics_binary = bics_binary.loc[:,passed_bics]
        print("%s biclusters not passed min_n_samples=%s filter removed:"%(len(not_passed_bics),min_n_samples),not_passed_bics)
    
    x, y = get_x_y(annot_treated,
                   attr_labels = surv_cols,
                   pos_label=1
                  )
    x = pd.concat([x, bics_binary.loc[x.index,:]],axis=1)
    return x,y

def run_RSF_and_calc_feature_importance(bics_train, annot_train,
                                        bics_test, annot_test,
                                        min_n_samples = 10, 
                                        treatment_col = "bevacizumab",
                                        treatment_value =1,
                                        covariates = [],
                                        surv_cols = ["_event","_time"],
                                        analysis_seed = 42,
                                        n_permutations = 100,
                                        param_grid = {}):
    
    """Fits RSF and performs permutation feature importance calculations
    evaluates prognostic values of biclusters in each feature group
    Returns feature importance and trained RSF modes
    """
    X_train, y_train = prepare_x_y_for_RSF(bics_train, annot_train,
                                           min_n_samples = min_n_samples , 
                                           treatment_col = treatment_col,
                                           treatment_value =treatment_value,
                                           covariates = covariates,
                                           surv_cols = surv_cols)
    X_test, y_test = prepare_x_y_for_RSF(bics_test, annot_test,
                                           min_n_samples = min_n_samples , 
                                           treatment_col = treatment_col,
                                           treatment_value =treatment_value,
                                           covariates = covariates,
                                           surv_cols = surv_cols)
    
    shared_features = sorted(set(X_train.columns.values).intersection(X_test.columns.values))
    X_train = X_train.loc[:,shared_features]
    X_test = X_test.loc[:,shared_features]

    # Initialize 5-fold CV GridSearchCV
    np.random.seed(analysis_seed)
    gs = GridSearchCV(RandomSurvivalForest(n_jobs=-1,random_state=analysis_seed,min_samples_split=10, min_samples_leaf=10),
                      param_grid,
                      cv=5, 
                      verbose=True, 
                      n_jobs=-1,
                      return_train_score=True)

    # Fit model
    gs.fit(X_train, y_train)

    # Best parameters and score
    print(f'Best parameters: {gs.best_params_}')
    #print(f'Best score: {gs.best_score_}') # mean CV score of the best_estimator 
    rsf = gs.best_estimator_

    print("RSF scores: Test: %.2f (train: %.2f)" % (rsf.score(X_test, y_test), rsf.score(X_train, y_train)))

    permut_result = permutation_importance(rsf, 
                                           X_test, 
                                           y_test,
                                           n_repeats=n_permutations, 
                                           n_jobs=-1,
                                           random_state=analysis_seed
                                          )

    permut_result = pd.DataFrame({ k: permut_result[k] for k in ("importances_mean","importances_std",)},
                 index=X_test.columns,
                ).sort_values(by="importances_mean", ascending=False)
    return permut_result, gs
