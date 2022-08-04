#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def AverageReward_plot(results, path, window_size = 1, ylim = (0, 500)):
    
    episodic_results = []
    for result in results:
        episodic_data = result.groupby(lambda x: int(x/500)).sum()
        episodic_results.append(episodic_data.rolling(window_size).mean())
    
    columns = episodic_results[0].columns
    mean = pd.DataFrame(np.mean(episodic_results, axis = 0), columns = columns)
    sd = pd.DataFrame(np.std(episodic_results, axis = 0), columns = columns)
    
    lb = mean - 2*sd
    ub = mean + 2*sd
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    mean.plot(kind = "line", y = columns, xlabel = "Online Episode", ylabel = "Average Reward", ax = ax, ylim = ylim)
    
    for i, col in enumerate(columns):
        ax.fill_between(sd.index, lb[col], ub[col], alpha = 0.3)
    
    fig.savefig(path, bbox_inches = "tight")
    plt.close(fig)
    
def CumulativeReward_plot(cumulative_results, sizes, path):
    
    algs = ["Rmax", "Vmax", "MoRmax"]
    for alg in algs:
        naive_df = pd.DataFrame()
        naive_df_sd = pd.DataFrame()
        controlled_df = pd.DataFrame()
        controlled_df_sd = pd.DataFrame()
        for i, result in enumerate(cumulative_results):
            mean = result[alg][0]
            sd = result[alg][1]
            naive_df["K = 0"] = mean["Ignore"]
            naive_df_sd["K = 0"] = sd["Ignore"]
            controlled_df["K = 0"] = mean["Ignore"]
            controlled_df_sd["K = 0"] = sd["Ignore"]
            naive_df[f"K = {sizes[i]}"] = mean["Naive"]
            naive_df_sd[f"K = {sizes[i]}"] = sd["Naive"]
            controlled_df[f"K = {sizes[i]}"] = mean["Backdoor"]
            controlled_df_sd[f"K = {sizes[i]}"] = sd["Backdoor"]
        
        ylim = (0, max(naive_df.to_numpy().max(), controlled_df.to_numpy().max())*1.1)
        
        columns = naive_df.columns
        
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        naive_df.plot(kind = "line", xlabel = "Online Episode", ylabel = "Cumulative Reward", ax = ax, ylim = ylim)
        naive_lb = naive_df - 2*naive_df_sd
        naive_ub = naive_df + 2*naive_df_sd
        for i, col in enumerate(columns):
            ax.fill_between(naive_df_sd.index, naive_lb[col], naive_ub[col], alpha = 0.3)
        fig.savefig(path + f"{alg}_naive_rewards.png", bbox_inches = "tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        controlled_df.plot(kind = "line", xlabel = "Online Episode", ylabel = "Cumulative Reward", ax = ax, ylim = ylim)
        controlled_lb = controlled_df - 2*controlled_df_sd
        controlled_ub = controlled_df + 2*controlled_df_sd
        for i, col in enumerate(columns):
            ax.fill_between(controlled_df_sd.index, controlled_lb[col], controlled_ub[col], alpha = 0.3)
        fig.savefig(path + f"{alg}_controlled_rewards.png", bbox_inches = "tight")
        plt.close(fig)
            
            
def get_cumulative_results(results):
    episodic_results = []
    for result in results:
        episodic_results.append(result.groupby(lambda x: int(x/500)).sum().cumsum())
    
    columns = episodic_results[0].columns
    mean = pd.DataFrame(np.mean(episodic_results, axis = 0), columns = columns)
    sd = pd.DataFrame(np.std(episodic_results, axis = 0), columns = columns)
    
    return [mean, sd]