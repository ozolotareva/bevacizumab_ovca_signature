import pandas as pd
import numpy as np
import os,sys

import seaborn as sns
import matplotlib.pyplot as plt

from lifelines.plotting import add_at_risk_counts
from lifelines import KaplanMeierFitter, CoxPHFitter

from statsmodels.stats.multitest import fdrcorrection
sys.path.append("unpast/")
from unpast.utils.method import cluster_samples, update_bicluster_data


### plotting KM for patients stratified by bicluster and treatment ###
def plot_KM_predictive(bic_samples, annot, surv_results, bic_id, direction,
                       surv_time = "_time",
                       surv_event= "_event",
                       cohort = "",
                       add_counts=False,
                       xlabel = "",
                       figsize =(10,2.5),
                       e = "expr.", # expression
                       xlim = False,
                       xticks=[12*x for x in range(0,9)],
                       label_pos =(0.05,0.1),
                       add_subplot=False):
    
    yticks = [0, 0.25,0.5,0.75, 1.0]
    sns.reset_defaults()
    sns.set(font_scale= 0.75,
            style='ticks',
            rc={'axes.edgecolor': '.5',
                'xtick.color': '.25',
                'ytick.color': '.25'})
    
    i = bic_id
    res = surv_results.loc[i,:]
    prefix = "bic"
    pv = surv_results.loc[i,prefix+".pval"]
    adjpv = surv_results.loc[i,prefix+".pval_BH"]
    hr = surv_results.loc[i,prefix+".HR"]
    ci95 = surv_results.loc[i,prefix+".CI95"]
    print(ci95)
    bic_text = "p-val.=%.2e\nadj.p-val.=%.2e\nHR=%.2f(%.2f,%.2f)"%(pv,adjpv, hr, ci95[0], ci95[1])
    print("bicluster",bic_text.replace("\n",", ")+";")

    prefix = "bg"
    pv = surv_results.loc[i,prefix+".pval"]
    adjpv = surv_results.loc[i,prefix+".pval_BH"]
    hr = surv_results.loc[i,prefix+".HR"]
    ci95 = surv_results.loc[i,prefix+".CI95"]
    print(ci95)
    bg_text = "p-val.=%.2e\nadj.p-val.=%.2e\nHR=%.2f(%.2f,%.2f)"%(pv,adjpv, hr, ci95[0], ci95[1])
    print("background",bg_text.replace("\n",", ")+";")

    bic = set(annot.index.values).intersection(bic_samples)
    bg = set(annot.index.values).difference(bic_samples)
    if direction=="DOWN":
        bic, bg = bg, bic
        bic_text, bg_text = bg_text, bic_text

    plt.figure(figsize=figsize)
    
    ax = plt.subplot(1,2,2)
    
    kmf_1 = KaplanMeierFitter()
    s1 = set(annot.loc[annot["bevacizumab"]==1,:].index.values).intersection(bic) 
    kmf_1.fit(annot.loc[s1,surv_time],
              annot.loc[s1,surv_event],
              label='bevacizumab, n=%s'%len(s1)
             ).plot_survival_function(ax=ax,
                                      color="red",
                                      xticks=xticks,
                                      yticks=yticks)

    kmf_2 = KaplanMeierFitter()
    s2 = set(annot.loc[annot["bevacizumab"]==0,:].index.values).intersection(bic) 
    kmf_2.fit(annot.loc[s2,surv_time],
              annot.loc[s2,surv_event],
              label='standard, n=%s'%len(s2)
             ).plot_survival_function(ax=ax,
                                      color="red",
                                      linestyle='dashed',
                                      xticks=xticks,
                                      yticks=yticks)

    ax.set_title("%s: Bicluster %s, high %s"%(cohort, i, e),fontsize=11)
    
    ax.set_xlabel(xlabel)
    ax.text(label_pos[0],label_pos[1], bic_text, fontsize=8)
    ax.set_ylim(0,1)
    ax.set_xlim(0,100)
    ax.legend(loc=1, prop={'size': 8})
    if add_counts:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf_1, kmf_2, ax=ax)
    
    ax = plt.subplot(1,2,1)

    kmf_3 = KaplanMeierFitter()
    s3 = set(annot.loc[annot["bevacizumab"]==1,:].index.values).intersection(bg) 
    kmf_3.fit(annot.loc[s3,surv_time],
              annot.loc[s3,surv_event],
              label='bevacizumab, n=%s'%len(s3)
             ).plot_survival_function(ax=ax,
                                      color="blue",
                                      xticks=xticks,
                                      yticks=yticks)

    kmf_4 = KaplanMeierFitter()
    s4 = set(annot.loc[annot["bevacizumab"]==0,:].index.values).intersection(bg) 
    kmf_4.fit(annot.loc[s4,surv_time],
              annot.loc[s4,surv_event],
              label='standard, n=%s'%len(s4)
             ).plot_survival_function(ax=ax,
                                      color="blue",
                                      linestyle='dashed',
                                      xticks=xticks,
                                      yticks=yticks) 
    print(i,len(bg))
    ax.set_title("%s: Bicluster %s, low %s"%(cohort, i, e),fontsize=11)
    ax.set_xlabel(xlabel)
    ax.text(label_pos[0],label_pos[1],bg_text,fontsize=8)
    ax.set_ylim(0,1)
    ax.set_xlim(0,100)
    ax.legend(loc=1, prop={'size': 8})
    if add_counts:
        tmp = add_at_risk_counts(kmf_3, kmf_4, ax=ax)
        plt.tight_layout()
        
    if not add_subplot:
        plt.show()

def plot_KM_prognostic(biclusters,
                       annot,
                       surv_results,
                       treatment = "bevacizumab",
                       surv_time='_time',
                       surv_event='_event',
                       cohort = "cohort",
                       xlabel = "months",
                       xticks=[12*x for x in range(0,9)],
                       add_counts=False,
                       label_pos =(2,0.05),
                       figsize=(5,2.5),
                       linestyle='solid',
                       add_subplot=False):
    yticks = [0, 0.25,0.5,0.75, 1.0]
    sns.reset_defaults()
    sns.set(font_scale= 0.75,
            style='ticks',
            rc={'axes.edgecolor':'.5',
                'xtick.color':'.25',
                'ytick.color':'.25'}
           )

    for i in biclusters.index.values:
        bic = set(biclusters.loc[i,"samples"]).intersection(set(annot.index.values))
        bg = set(annot.index.values).difference(bic)
        bic = sorted(bic)
        bg = sorted(bg)

        if len(biclusters.loc[i,"genes_up"]) >= len(biclusters.loc[i,"genes_down"]):
            up, down = bic, bg
        else:
            up, down = bg, bic

        print(i,annot.shape[0],len(up),len(down))

        pv = surv_results.loc[i,"pval"]
        adjpv = surv_results.loc[i,"adj_pval"]
        hr = surv_results.loc[i,"HR"]
        ci95 = surv_results.loc[i,["CI_l","CI_r"]]
        text = "p-val.=%.2e\nadj.p-val.=%.2e\nHR=%.2f(%.2f,%.2f)"%(pv,adjpv, hr, ci95[0], ci95[1])
        
        #plt.figure(figsize=figsize)
        #ax = plt.subplots(111)
        fig, ax= plt.subplots(1, 1, figsize=figsize)
        ax.set_xticks(xticks)

        kmf_1 = KaplanMeierFitter()

        ax = kmf_1.fit(annot.loc[up,surv_time],
                       annot.loc[up,surv_event],
                       label='high (n=%s)'%len(up)
                      ).plot_survival_function(ax=ax, 
                                               color="red",
                                               linestyle=linestyle,
                                               xticks =xticks,
                                               yticks =yticks)

        kmf_2 = KaplanMeierFitter()

        ax = kmf_2.fit(annot.loc[down,surv_time], 
                       annot.loc[down,surv_event],
                       label='low (n=%s)'%len(down)
                      ).plot_survival_function(ax=ax,
                                               color="blue",
                                               linestyle=linestyle,
                                               xticks =xticks,
                                               yticks =yticks) 
        
        if add_counts:
            #from lifelines.plotting import add_at_risk_counts
            add_at_risk_counts(kmf_1, kmf_2, ax=ax)
            ax.set_xlabel(ax.get_xlabel(),
                          ha = "right",
                          fontsize=9
                         )
            plt.tight_layout()
        ax.set_title("%s, %s (n=%s), Bicluster %s"%(cohort, 
                                                    treatment,
                                                    len(up)+len(down),
                                                    i)
                    )
        ax.legend(loc=1, prop={'size': 8})
        ax.set_xlabel(xlabel)
        ax.text(label_pos[0],label_pos[1],text,fontsize=8)
        ax.set_ylim(0,1)
        ax.set_xlim(0,100)
        
        if not add_subplot:
            plt.show()                   
            
def plot_KM_four(bic_samples,
                 annot,
                 bic_id,
                 direction,
                 figsize=(5,4.5),
                 xticks=[12*x for x in range(0,9)],
                 xlabel = "",#"%s, months"%s, # ""
                 ax = "",
                 surv_time = "OS_time",
                 surv_event= "OS_event",
                 font_scale=0.85,
                 cohort = "",
                 add_counts=False,
                 label_pos =(0.05,0.1),
                 add_subplot=False,
                 ci_show=False
                ):    
    
    i = bic_id
    if direction == "UP":
        d1,d2 = "high", "low"
        c1,c2 = "red","blue"
    elif direction == "DOWN":
        d1,d2 = "low", "high"
        c1,c2 = "blue","red"
    else:
        d1,d2 = "bic.", "bg."
        c1,c2 = "green","grey"
        
    bic = set(annot.index.values).intersection(bic_samples)
    bg = set(annot.index.values).difference(bic_samples)

    
    yticks = [0, 0.25,0.5,0.75, 1.0]
    sns.reset_defaults()
    sns.set(font_scale= font_scale,
            style='ticks',
            rc={'axes.edgecolor': '.5',
                'xtick.color': '.25',
                'ytick.color': '.25'})

    if not ax:
        plt.figure(figsize=figsize)
        ax = plt.subplot(1,1,1)

    kmf_1 = KaplanMeierFitter()
    s1 = set(annot.loc[annot["bevacizumab"]==1,:].index.values).intersection(bic) 
    kmf_1.fit(annot.loc[s1,surv_time], 
              annot.loc[s1,surv_event], 
              label='bevacizumab, %s, n=%s'%(d1,len(s1))
             ).plot_survival_function(ax=ax, color=c1,
                                      xticks=xticks,
                                      yticks=yticks,
                                      ci_show=ci_show)

    kmf_2 = KaplanMeierFitter()
    s2 = set(annot.loc[annot["bevacizumab"]==0,:].index.values).intersection(bic) 
    kmf_2.fit(annot.loc[s2,surv_time],
              annot.loc[s2,surv_event],
              label='standard, %s, n=%s'%(d1,len(s2))
             ).plot_survival_function(ax=ax, color=c1,
                                      linestyle='dashed',
                                      xticks=xticks,
                                      yticks=yticks,
                                      ci_show=ci_show)

    kmf_3 = KaplanMeierFitter()
    s3 = set(annot.loc[annot["bevacizumab"]==1,:].index.values).intersection(bg) 
    kmf_3.fit(annot.loc[s3,surv_time],
              annot.loc[s3,surv_event],
              label='bevacizumab, %s, n=%s'%(d2,len(s3))
             ).plot_survival_function(ax=ax, color=c2,
                                     xticks=xticks,
                                      yticks=yticks,
                                      ci_show=ci_show)

    kmf_4 = KaplanMeierFitter()
    s4 = set(annot.loc[annot["bevacizumab"]==0,:].index.values).intersection(bg) 
    kmf_4.fit(annot.loc[s4,surv_time],
              annot.loc[s4,surv_event],
              label='standard, %s, n=%s'%(d2,len(s4))
             ).plot_survival_function(ax=ax,
                                      color=c2,
                                      linestyle='dashed',
                                     xticks=xticks,
                                      yticks=yticks,
                                      ci_show=ci_show) 
    ax.set_xlabel(xlabel)
    ax.set_ylim(0,1)
    ax.set_xlim(0,100)
    
    tmp = ax.set_title("%s (n=%s): Bicluster %s"%(cohort,annot.shape[0],bic_id),fontdict={'size':10})
    
    
    if add_counts:
        from lifelines.plotting import add_at_risk_counts
        tmp = add_at_risk_counts(kmf_1, kmf_2,kmf_3, kmf_4,
                                 ax=ax, xticks=xticks, #fig=fig,
                                 ypos=-0.35,fontdict={'size':8})
        tmp = plt.tight_layout()
    if not add_subplot:
        plt.show()