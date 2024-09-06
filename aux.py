import pandas as pd
import numpy as np
import os,sys

import seaborn as sns
import matplotlib.pyplot as plt

from lifelines.plotting import add_at_risk_counts
from lifelines import KaplanMeierFitter, CoxPHFitter

def zscore(df):
    m = df.mean(axis=1)
    df = df.T - m
    df = df.T
    s = df.std(axis=1)
    df = df.T/s
    df = df.T
    # set to 0 not variable genes 
    zero_var_genes = s[s==0].index.values
    if len(zero_var_genes)>0:
        print(len(zero_var_genes),"zero variance rows detected, assign zero z-scores ",file = sys.stderr)
    df.loc[zero_var_genes,:] = 0
    return df

def compare_clusterings(true_labels,pred_labels):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.cluster import pair_confusion_matrix
    stat = {}
    for k1 in set(true_labels.values):
        stat[k1]={}
        for k2 in set(pred_labels.values):
            stat[k1][k2] = len(set(true_labels[true_labels==k1].index).intersection(set(pred_labels[pred_labels==k2].index)))
    conf_matrix = pd.DataFrame.from_dict(stat)
    
    pcm = pair_confusion_matrix(true_labels,pred_labels)
    corr = pcm[0,0]+pcm[1,1]
    incorr = pcm[0,1]+pcm[1,0]
    total_pairs = len(true_labels)*(len(true_labels)-1)

    print("correct pairs:%.2f ,"
          "incorrect pairs:%.2f"%(corr/total_pairs*100,
                                  incorr/total_pairs*100))
    ARI = adjusted_rand_score(true_labels,pred_labels)
    print("ARI:%.2f"%ARI)
    return conf_matrix


def plot_subtype_heatmap(exprs, subt, biomarkers,
                         color_dict ={}, gene_id = "gene",
                        figsize=(5,5)):
    
    color_order = ["lime","orange","cyan","magenta"]
    sns.set(font_scale=1.5)
    shared_genes = [x for x in biomarkers[gene_id].values if x in exprs.index]
    not_found = [x for x in biomarkers[gene_id].values if x not in exprs.index]
    if len(not_found)>0:
        print("not found in expression:",not_found)
    e = exprs.loc[shared_genes,:]
    
    col_colors = []
    for cl in ["consensusOV"]: #"consensusOV","Bentink"]:
        col_color = subt.loc[:,[cl]]
        col_color[cl] =  col_color[cl].apply(lambda x: color_dict[cl][x] )
        col_colors.append(col_color)
    col_colors = pd.concat(col_colors,axis=1)

    cl = 'subtype'
    row_colors = biomarkers.set_index(gene_id)
    row_colors = row_colors.loc[shared_genes,[cl]]
    row_colors[cl]= row_colors[cl].apply(lambda x:  color_dict[cl][x] )

    cl = "consensusOV"
    e_sorted = []
    for c in color_order:
        e_sorted.append(e.loc[:,col_colors.loc[col_colors[cl]==c,:].index.values])
    e = pd.concat(e_sorted,axis=1)

    cl = 'subtype'
    e_sorted = []
    for c in color_order:
        e_sorted.append(e.loc[row_colors.loc[row_colors[cl]==c,:].index.values,:])
    e = pd.concat(e_sorted,axis=0)
    
    # color annotation labels
    row_colors.columns = [""]
    col_colors.columns = [""]
    
    g = sns.clustermap(zscore(e), vmin=-3, vmax=3,figsize=figsize,
                   cmap=sns.color_palette("coolwarm", as_cmap=True),
                   col_colors=col_colors, col_cluster=False,
                    row_colors=row_colors, row_cluster=False,
                       xticklabels=False, yticklabels=False,
                       dendrogram_ratio=(0.01,0.01)
                  )
    
    return g

def plot_KM_predictive_subtypes(annot,
                        covariates = [],
                        t1= "bevacizumab", t0="standard",
                        xlabel = "", # surv+", months"
                        xlim = "",
                        title = "",
                        time_col="time_column", event_col="event_column",
                        treatment_col= "bevacizumab",
                        subt_col = "ConsensusOV_subt" ,subts= ["PRO","MES","DIF","IMR"],
                        label_pos=(40,0.6),
                        add_text = True):
    sns.reset_defaults()
    plt.figure(figsize=(15,3))
    stats = {}
    i = 1
    for s in subts: #["proliferative","mesenchymal","differentiated","immunoreactive"]:
        a = annot.loc[annot[subt_col ].str.contains(s),[treatment_col,time_col,event_col]+covariates ]
        s1 = set(a.loc[a[treatment_col]==1,:].index.values)
        s2 = set(a.loc[a[treatment_col]==0,:].index.values)

        a = a.loc[sorted(s1|s2),:]

        cph = CoxPHFitter()
        res = cph.fit(a, duration_col=time_col, event_col= event_col, 
                      show_progress=False, formula="age + FIGO_IIIC + FIGO_IV + OP_1 + OP_2 + "+treatment_col)

        res_predictive = res.summary.loc[treatment_col,["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]]
        pval = res.summary.loc[treatment_col,"p"]
        hr = res.summary.loc[treatment_col,"exp(coef)"]
        CI_l = res.summary.loc[treatment_col,"exp(coef) lower 95%"]
        CI_r = res.summary.loc[treatment_col,"exp(coef) upper 95%"]
        subt_text= "p-val.=%.2e\nHR=%.2f"%(pval, hr)
        #print(s, subt_text, "CI95=(%.2f,%.2f)"%(CI_l, CI_r))
        
        stats[s] = {t1:len(s1),
                    t0:len(s2),
                   "HR (CI95%)":"%.2f (%.2f-%.2f)"%(hr,CI_l,CI_r),
                   "p-value":"%.2f"%pval}


        ax = plt.subplot(1,4,i)

        kmf_1 = KaplanMeierFitter()

        ax = kmf_1.fit(a.loc[s1,time_col], a.loc[s1,event_col], 
                       label=t1).plot_survival_function(ax=ax, color="green") # label=t1+", n=%s"%len(s1)

        kmf_2 = KaplanMeierFitter()

        ax = kmf_2.fit(a.loc[s2,time_col], a.loc[s2,time_col],
                       label=t0).plot_survival_function(ax=ax, color="grey") #, linestyle='dashed') # label=t0+", n=%s"%len(s2)

        if i>1:
            ax.get_legend().remove()
        else:
            pass# legend transparent
        
        if xlabel: 
            ax.set_xlabel(xlabel)
        if add_text:
            ax.text(label_pos[0],label_pos[1], subt_text)
        ax.set_ylim(0,1)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_title("%s(%s;n=%s)" %(s,title,a.shape[0]))
        #add_at_risk_counts(kmf_1, kmf_2, ax=ax) 
        #plt.tight_layout()
        i += 1
    return pd.DataFrame.from_dict(stats).T


def plot_KM_prognostic_subytpes(annot, 
                        surv_event="event_column",surv_time="time_column",
                        xlabel="survival, months", 
                        title = "",
                        covariates = [],
                        subtypes = ["proliferative","mesenchymal","differentiated","immunoreactive"],
                        color_dict = {},
                        cohort_name = "cohort",
                        figsize=(5,3),plot_legend = True
                       ):
    stats = {}
    sns.reset_defaults()
    plt.figure(figsize=figsize)
    
    ax = plt.subplot(111)
    for subt in subtypes:

        s1 = set(annot.loc[annot["subtype"]==subt,:].index.values)
        a = annot.loc[:,[surv_event,surv_time]+covariates].dropna()
        s1 = s1.intersection(a.index.values)
        a[subt] = 0
        a.loc[sorted(s1),subt]=1

        cph = CoxPHFitter()
        res = cph.fit(a, duration_col=surv_time, event_col= surv_event, 
                      show_progress=False) # , formula="age + FIGO_IIIC + FIGO_IV + OP_1 + OP_2 + "+subt

        res_predictive = res.summary.loc[subt,["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]]
        pval = res.summary.loc[subt,"p"]
        hr = res.summary.loc[subt,"exp(coef)"]
        CI_l = res.summary.loc[subt,"exp(coef) lower 95%"]
        CI_r = res.summary.loc[subt,"exp(coef) upper 95%"]

        stats[subt] = {"n_samples":len(s1),
                   "HR (CI95%)":"%.2f (%.2f-%.2f)"%(hr,CI_l,CI_r),
                   "p-value":"%.2f"%pval}


        kmf = KaplanMeierFitter()
        ax = kmf.fit(a.loc[s1,surv_time], a.loc[s1,surv_event],
                     label=subt).plot_survival_function(ax=ax, color=color_dict["subtype"][subt])
    if title:
        plt.title(title)
    #add_at_risk_counts(kmf_1, kmf_2, ax=ax)
    ax.set_xlabel(xlabel)
    if not plot_legend:
        ax.get_legend().remove()

    return pd.DataFrame.from_dict(stats).T


def fitCPH_and_plotKM_treatments(annot,
                      surv_event="OS_event",
                      surv_time="OS_time",
                      covariates=[],
                      title="",
                      figsize=(5,4.5),
                      xticks=[12*x for x in range(0,9)],
                      max_time=100,
                      label_x_pos=2,
                      label_y_pos=0.05,
                      add_at_risk_counts = True
                      ):
    
    
    yticks = [0, 0.25,0.5,0.75, 1.0]
    a = annot.loc[:,[surv_event,surv_time]+covariates].dropna()
    cph = CoxPHFitter()
    
    res = cph.fit(a,
                  duration_col=surv_time,
                  event_col= surv_event, 
                  show_progress=False) 
    
    cols =  ["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]
    hr, CI_l, CI_r, pval = res.summary.loc["bevacizumab",cols].values
    
    print("p-value=%.2f\nHR=%.2f(%.2f-%.2f)"%(pval, hr, CI_l, CI_r, ))
    if pval <0.05:
        text = "p-value=%.2e\nHR=%.2f(%.2f-%.2f)"%(pval, hr, CI_l, CI_r, )
    else:
        text = "p-value=%.2f"%(pval)
    
    sns.reset_defaults()
    sns.set(font_scale=0.85,
            style='ticks',
            rc={'axes.edgecolor': '.5',
                'xtick.color': '.25',
                'ytick.color': '.25'})
    tmp = plt.figure(figsize=figsize)

    ax = plt.subplot(111)
    kmf1 = KaplanMeierFitter()
    ax = kmf1.fit(a.loc[a["bevacizumab"]==1,surv_time], a.loc[a["bevacizumab"]==1,surv_event],
                     label="bevacizumab").plot_survival_function(ax=ax, color="darkgreen",
                                                                 xticks=xticks,
                                                                 yticks=yticks)
    kmf2 = KaplanMeierFitter()
    ax = kmf2.fit(a.loc[a["bevacizumab"]==0,surv_time], a.loc[a["bevacizumab"]==0,surv_event],
                     label="standard").plot_survival_function(ax=ax, color="grey",
                                                              xticks=xticks,
                                                              yticks=yticks)
    plt.title(title,fontdict={'size':11})
    if add_at_risk_counts:
        add_at_risk_counts(kmf1, kmf2, ax=ax,
                       xticks=xticks)
        tmp = ax.set_xlabel(None)
        tmp = ax.set_xlim(0,max_time)
        tmp = ax.text(label_x_pos,label_y_pos, text,fontdict={'size':9})
        tmp = ax.set_ylim(0,1)
        plt.tight_layout()
    plt.show()    
