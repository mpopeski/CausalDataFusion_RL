#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool

from Rmax import R_MAX
from Vmax import Vmax
from MoRmax import MoRmax
from MDP_environments import TabularMDP

from plotting import AverageReward_plot, CumulativeReward_plot, get_cumulative_results


def main(K_obs):
    data = env.get_obs_data(K_obs)    
    
    integration = ["ignore", "naive", "controlled"]
    integration_index = ["Ignore", "Naive", "Backdoor"] 
    
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    
    path_ = path + f"{K_obs}/" 
    cumulative_results = {}
    
    path_Rmax = path_ + "Rmax/"
    os.makedirs(path_Rmax + "Figures/", exist_ok=True)
    path_Rmax_avgRew = path_Rmax + "Figures/Rmax_exp1.png"
    results_Rmax = []
    for rep in range(reps):
        result = []
        for integ in integration:
            model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
            model1.initialize(data, integ)
            model1.learn()
            result.append(model1.reward)
        result = pd.DataFrame(result, index = integration_index).T
        result.to_csv(path_Rmax + f"results{rep}.csv")
        results_Rmax.append(result)
    AverageReward_plot(results_Rmax, path_Rmax_avgRew)
    cumulative_results["Rmax"] = get_cumulative_results(results_Rmax)
    
    path_Vmax = path_ + "Vmax/"
    os.makedirs(path_Vmax + "Figures/", exist_ok=True)
    path_Vmax_avgRew = path_Vmax + "Figures/Vmax_exp1.png"
    results_Vmax = []
    for rep in range(reps):
        result = []
        for integ in integration:
            model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
            model2.initialize(data, integ)
            model2.learn()
            result.append(model2.reward)
        result = pd.DataFrame(result, index = integration_index).T
        result.to_csv(path_Vmax + f"results{rep}.csv")
        results_Vmax.append(result)
    AverageReward_plot(results_Vmax, path_Vmax_avgRew)
    cumulative_results["Vmax"] = get_cumulative_results(results_Vmax)
    
    path_MoRmax = path_ + "MoRmax/"
    os.makedirs(path_MoRmax + "Figures/", exist_ok=True)
    path_MoRmax_avgRew = path_MoRmax + "Figures/MoRmax_exp1.png"
    results_MoRmax = []
    for rep in range(reps):
        result = []
        for integ in integration:
            model3 = MoRmax(env, gamma, m, eta, Rmax, K_int)
            model3.initialize(data, integ)
            model3.learn()
            result.append(model3.reward)
        result = pd.DataFrame(result, index = integration_index).T
        result.to_csv(path_MoRmax + f"results{rep}.csv")
        results_MoRmax.append(result)
    AverageReward_plot(results_MoRmax, path_MoRmax_avgRew, window_size=3)
    cumulative_results["MoRmax"] = get_cumulative_results(results_MoRmax)
    
    #data["r"].to_csv(path_ + "obs_rew.csv")
    
    return cumulative_results



if __name__ == "__main__":
    
    env = TabularMDP(5, 0, 500, default_prob = 4, n_reward_states = 12, policy = "v3_eng", simpson = True)
    load_path = "./Environment_exp1/"
    path = "../Clean_repeat/Results_exp1/"
    
    if load_path:
        env.load_env(load_path)
    else:        
        env.save_env(path + "Environment/")
    
    # parameters for all alorithms
    m = 1000
    K_int = 300
        
    sizes = [500, 1000, 2500, 5000]

    print("starting to learn different models")
    with Pool(processes=None) as p:
        cumulative_results = p.map(main, sizes)
    
    CumulativeReward_plot(cumulative_results, sizes, path)