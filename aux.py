import pandas as pd
import numpy as np
import os,sys

import seaborn as sns
import matplotlib.pyplot as plt

from lifelines.plotting import add_at_risk_counts
from lifelines import KaplanMeierFitter, CoxPHFitter

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
    
    
