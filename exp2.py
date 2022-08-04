#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool

from Rmax import R_MAX
from Vmax import Vmax
from MoRmax import MoRmax
from MDP_environments import TabularMDP

from plotting import AverageReward_plot


def main(conf_val):
    path_ = path + f"{conf_val}/" 

    os.makedirs(path_, exist_ok=True)

    env = TabularMDP(state_values = 5, action_values = 3, H = 500, default_prob = 4, n_reward_states=12, policy = "random", 
                     simpson = False, conf_values = conf_val)
    
    if load_path:
        load_path_ = load_path + f"{conf_val}/Environment/"
        env.load_env(load_path_)
    else:
        env.save_env(path_ + "Environment/")
    
    print("collecting observational data")
    data = env.get_obs_data(K_obs)   
    
    integration = ["ignore", "controlled", "controlled_FD"]
    integration_index = ["Ignore", "Backdoor", "Frontdoor"] 
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
        
    path_Rmax = path_ + "Rmax/"
    os.makedirs(path_Rmax + "Figures/", exist_ok=True)
    path_Rmax_avgRew = path_Rmax + "Figures/Rmax_exp2.png"
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
    AverageReward_plot(results_Rmax, path_Rmax_avgRew, window_size = 15, ylim = (0,150))
    
    path_Vmax = path_ + "Vmax/"
    os.makedirs(path_Vmax + "Figures/", exist_ok=True)
    path_Vmax_avgRew = path_Vmax + "Figures/Vmax_exp2.png"
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
    AverageReward_plot(results_Vmax, path_Vmax_avgRew, window_size = 15, ylim = (0, 150))
    
    path_MoRmax = path_ + "MoRmax/"
    os.makedirs(path_MoRmax + "Figures/", exist_ok=True)
    path_MoRmax_avgRew = path_MoRmax + "Figures/MoRmax_exp2.png"
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
    AverageReward_plot(results_MoRmax, path_MoRmax_avgRew, window_size = 15, ylim = (0, 150))


if __name__ == "__main__":

    path = "../Results_exp2/"
    load_path = "./Environments_exp2/"
    K_obs = 5000 
    
    m = 1000
    K_int = 500
    
    conf_vals = [4,8,16,32,64]
    with Pool(processes=None) as p:
        p.map(main, conf_vals)
        
