import pandas as pd
import numpy as np
import os,sys

import seaborn as sns
import matplotlib.pyplot as plt

from lifelines.plotting import add_at_risk_counts
from lifelines import KaplanMeierFitter, CoxPHFitter

from statsmodels.stats.multitest import fdrcorrection

from unpast.utils.method import cluster_samples, update_bicluster_data

#### survival analysis for biclusters  #####
def test_predictive_biclusters(bics,annot, min_n_samples = 10,
                           covariates = [], 
                           treatment_col = "bevacizumab", 
                           time_col = "_time", 
                           event_col = "_event"):

    # bics - dataframe with "samples" column, where samples is a set of sample ids
    # annot - sample metadata for survival analysis
    # min_n_samples  - minimal number of samples in treatment x biomarker group

    result = {}

    for i in bics.index.values:
        result[i]={}
        bic = bics.loc[i,"samples"]

        bic = set(annot.index.values).intersection(bic)
        bg = set(annot.index.values).difference(bic)
        
        t0 = set(annot.loc[annot[treatment_col]==0,:].index.values)
        t1 = set(annot.loc[annot[treatment_col]==1,:].index.values)
        
        if min(len(bic&t0),len(bg&t0),len(bic&t1),len(bg&t1))<min_n_samples:
            print(i, "not tested as some sample groups are too small:", len(bic&t0),len(bg&t0),len(bic&t1),len(bg&t1))
        else:
            ### Background ### 
            bg_annot = annot.loc[sorted(bg),
                                 [treatment_col]+covariates+[time_col, event_col]].copy()
            
            hr, ci, pval, covar_excl = test_samples_predictive(bg_annot,
                            bg, # compare treatments in this sample group
                            covariates = covariates,
                            treatment_col = treatment_col,
                            time_col =time_col,
                            event_col =event_col
                           )
            
            result[i].update({"bg.pval":pval,
                              "bg.HR":hr,
                              "bg.CI95" : ci,
                              "bg.beva":bg_annot.loc[bg_annot[treatment_col]==1,:].shape[0],
                              "bg.st":bg_annot.loc[bg_annot[treatment_col]==0,:].shape[0],
                              "bg.covar_excl":covar_excl
                             })
            
            ### Bicluster ### 
            bic_annot = annot.loc[sorted(bic),:].copy()
            bic_annot = bic_annot.loc[:,[treatment_col]+covariates+[time_col, event_col]]
            hr, ci, pval, covar_excl = test_samples_predictive(bic_annot,
                            bic, # compare treatments in this sample group
                            covariates = covariates,
                            treatment_col = treatment_col,
                            time_col =time_col,
                            event_col =event_col
                           )
            result[i].update({"bic.pval":pval,
                              "bic.HR":hr,
                              "bic.CI95" : ci,
                              "bic.beva":bic_annot.loc[bic_annot[treatment_col]==1,:].shape[0],
                              "bic.st":bic_annot.loc[bic_annot[treatment_col]==0,:].shape[0],
                              "bic.covar_excl":covar_excl
                             })
            result[i]["direction"] = bics.loc[i,"direction"]
            
            # decide if low or high expression is predicitve
            result[i]["predictive"] = "high"
            if len(bics.loc[i,"genes_down"]) > len(bics.loc[i,"genes_up"]):
                result[i]["predictive"] = "low"
            if result[i]["bg.pval"] < result[i]["bic.pval"]:
                if result[i]["predictive"] == "high":
                    result[i]["predictive"] = "low"
                elif result[i]["predictive"] == "low":
                    result[i]["predictive"] = "high"
            
            # decide effect for bevacizumab
            result[i][treatment_col+"_eff"] = result[i]["bic.HR"]
            if result[i]["bg.pval"] < result[i]["bic.pval"]:
                result[i][treatment_col+"_eff"] = result[i]["bg.HR"]
            
            if result[i][treatment_col+"_eff"]<1:
                result[i][treatment_col+"_eff"] ="pos"
            else:
                result[i][treatment_col+"_eff"] ="neg"
            
    df = pd.DataFrame.from_dict(result).T
    #df = df.dropna()
    print("Biclusters tested: %s"%df.shape[0])
    bh_res, adj_pval = fdrcorrection(df["bic.pval"].values, alpha=0.05)
    df["bic.pval_BH"] = adj_pval
    df = df.sort_values(by = "bg.pval")
    bh_res, adj_pval = fdrcorrection(df["bg.pval"].values, alpha=0.05)
    df["bg.pval_BH"] = adj_pval
    
    df["min_adj_pval"] = df.loc[:,["bic.pval_BH","bg.pval_BH"]].min(axis=1)
    df["min_pval"] = df.loc[:,["bic.pval","bg.pval"]].min(axis=1)
    df = df.sort_values(by=["min_adj_pval","min_pval"])
    return df

def fit_cph(a, surv_time, event, target_col="x"):
    cph = CoxPHFitter()
    try:
        cph.fit(a, duration_col=surv_time, event_col= event, show_progress=False)
        results = cph.summary  
        pval = results.loc[target_col,"p"]
        hr = results.loc[target_col,"exp(coef)"]
        upper_95CI = results.loc[target_col,"exp(coef) upper 95%"]
        lower_95CI = results.loc[target_col,"exp(coef) lower 95%"]
        return hr,lower_95CI, upper_95CI,pval
    except:
        return np.nan,np.nan,np.nan,np.nan

def check_surv_data(surv_data, event,time, target_column = "x",verbose=True):
    cov_kept = []
    events = surv_data[event].astype(bool)
    
    v1 = surv_data.loc[events, target_column].var()
    v2 = surv_data.loc[~events, target_column].var()
    
    v3 = surv_data.loc[surv_data[target_column]==1, event].var()
    v4 = surv_data.loc[surv_data[target_column]==0, event].var()
    
    if v1 ==0 or v2 ==0:
        if verbose:
            in_bic = surv_data.loc[surv_data[target_column]==1,:].shape[0]
            in_bg = surv_data.loc[surv_data[target_column]==0,:].shape[0]
            print("perfect separation for target group of  %s/%s samples"%(in_bic,in_bg),
                 "variances: {:.2f} {:.2f}".format(v1, v2), file = sys.stderr)
    if v3 == 0 and verbose:
        print("zero variance for events in group; all events are ",set(surv_data.loc[surv_data[target_column]==1, event].values))
    if v4 == 0 and verbose:
        print("zero variance for events in background; all events are ",set(surv_data.loc[surv_data[target_column]==0, event].values))
    
    # check variance of covariates in event groups
    exclude_covars =[]
    for c in [x for x in surv_data.columns.values if not x in [target_column,event, time]]:
        if surv_data.loc[events, c].var()==0:
            exclude_covars.append(c)
            if verbose:
                print("\t",c,"variance is low in event group, c=",set(surv_data.loc[events, c].values),file = sys.stdout)
        if surv_data.loc[~events, c].var()==0:
            exclude_covars.append(c)
            if verbose:
                print("\t",c,"variance is low in no-event group, c=",set(surv_data.loc[~events, c].values),file = sys.stdout)
    if len(exclude_covars)>0:
        cols = surv_data.columns.values
        cols = [x for x in cols if not x in exclude_covars]
        surv_data = surv_data.loc[:,cols]
        if verbose:
            print("\t exclude covariates:",exclude_covars,"keep:",cols)
    cov_kept = [ x for x in surv_data.columns if x not in [event, time, target_column]]
    return surv_data, cov_kept

def test_samples_predictive(sample_data,
                            samples, # compare treatments in this sample group
                            covariates = [],
                            treatment_col = "bevacizumab",
                            time_col = "_time",
                            event_col = "_event"
                           ):
    surv_data = sample_data.copy()
    cols = [treatment_col]+covariates+[time_col, event_col]
    surv_data = surv_data.loc[samples,cols]
    surv_data,covar_kept = check_surv_data(surv_data,
                                           event =  event_col,
                                           time=time_col,
                                           target_column = treatment_col,
                                           verbose=False)
    
    hr,l95CI, u95CI, pval = fit_cph(surv_data,
                                    time_col,
                                    event_col,
                                   target_col = treatment_col)
    ci = (l95CI, u95CI)
    covar_excluded = [x for x in covariates if not x in covar_kept]
    return hr, ci, pval, covar_excluded

def test_predictive_with_interaction(bics,
                                     surv_data,
                                     covariates=[],
                                     treatment_col = "bevacizumab",
                                     duration_col="_time",
                                     event_col= "_event",
                                     min_n_samples=10,
                                     
                                    ):
    surv_results = {}
    
    formula= " + ".join(covariates)+" + "+ treatment_col+" * bic"
    print("formula:",formula)
    
    for i in bics.index.values:
        surv_results[i]={}
        bic = bics.loc[i,"samples"]
        d = bics.loc[i,"direction"]

        bic = set(surv_data.index.values).intersection(bic)
        bg = set(surv_data.index.values).difference(bic)
        
        t0 = set(surv_data.loc[surv_data[treatment_col]==0,:].index.values)
        t1 = set(surv_data.loc[surv_data[treatment_col]==1,:].index.values)
        
        surv_results[i].update({treatment_col+"_bic":len(bic&t1),
                               treatment_col+"_bg":len(bg&t1),
                               "standard"+"_bic":len(bic&t0),
                               "standard"+"_bg":len(bg&t0)})
        if min(len(bic&t0),len(bg&t0),len(bic&t1),len(bg&t1))<min_n_samples:
            print(i, "not tested as some sample groups are too small:", len(bic&t0),len(bg&t0),len(bic&t1),len(bg&t1))
        else:
            surv_data_bic = surv_data.copy()
            surv_data_bic["bic"] = 0
            surv_data_bic.loc[sorted(bic.intersection(set(surv_data.index.values))),"bic"] = 1
            cph = CoxPHFitter()
            res = cph.fit(surv_data_bic,
                          duration_col=duration_col,
                          event_col= event_col, 
                          show_progress=False,
                          formula=formula).summary
            for v in ["bevacizumab:bic","bevacizumab","bic"]: #
                surv_results[i][v+"_pval"] = res.loc[v,"p"]
                surv_results[i][v+"_HR"] = res.loc[v,"exp(coef)"]
                surv_results[i][v+"_CI95"] = (res.loc[v,"exp(coef) upper 95%"],res.loc[v,"exp(coef) lower 95%"])

    surv_results = pd.DataFrame.from_dict(surv_results).T
    surv_results = surv_results.dropna()
    for v in ["bevacizumab:bic","bevacizumab","bic"]: #
        surv_results = surv_results.sort_values(by=v+"_pval")
        bh_res, adj_pval = fdrcorrection(surv_results[v+"_pval"].values, alpha=0.05, is_sorted=False)
        surv_results[v+"_pval_BH"] = adj_pval
    
    return surv_results

def test_biclusters_prognostic(biclusters,annot, 
                    surv_event="_event",surv_time = "_time",
                   covariates = [],min_n_samples=10):
    """High-exprs. vs low-expression"""
    stats = {}
    for i in biclusters.index.values:
        a = annot.loc[:,[surv_event,surv_time]+covariates]
        s_bic = biclusters.loc[i,"samples"]
        a_bic = s_bic.intersection(set(a.index.values))
        a["bicluster"] = 0
        a.loc[sorted(a_bic),"bicluster"] = 1
        n_up = len(a_bic)
        n_down = len(set(a.index.values).difference(a_bic))
        # flip down-reg biclsuters 
        if biclusters.loc[i,"direction"] =="DOWN":
            a["bicluster"] = 1
            a.loc[sorted(a_bic),"bicluster"] = 0
            n_up, n_down =  n_down, n_up
        stats[i] = {"n_up":n_up,"n_down":n_down}
        if min(n_up, n_down) >= min_n_samples:
            a, passed_cov = check_surv_data(a, 
                                            surv_event,
                                            surv_time,
                                            target_column = "bicluster",
                                            verbose=False)
            if passed_cov!= covariates:
                print("bicluster",i,"covariates=",passed_cov)
            cph = CoxPHFitter()
            res = cph.fit(a, duration_col=surv_time, event_col= surv_event, 
                                  show_progress=False) 
            hr, CI_l, CI_r, pval = res.summary.loc["bicluster", ["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].values
            #print("bic:%s\tHR=%.2f (%.2f-%.2f) p-val = %s"%(i, hr, CI_l, CI_r, pval))
            stats[i].update({"HR":hr,"CI_l":CI_l,"CI_r":CI_r,"pval":pval})
    stats = pd.DataFrame.from_dict(stats).T.dropna()
    bh_res, adj_pval = fdrcorrection(stats["pval"].fillna(1).values, alpha=0.05)
    stats["adj_pval"] =  adj_pval 
    stats = stats.sort_values(by = "pval")
    return stats