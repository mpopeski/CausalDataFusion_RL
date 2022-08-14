#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from multiprocessing import Pool

import yaml
import pandas as pd

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
    
    parser = argparse.ArgumentParser(description="Training Params")
    parser.add_argument('config', type = str, help='Path to a config file')
    _args = parser.parse_args()
    
    with open(_args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    path = cfg.get("save_path", "./Results/exp2/")
    load_path = cfg.get("load_path", "")
    
    m = cfg.get("m", 100)
    K_obs = cfg.get("K_obs", 500)
    K_int = cfg.get("K_int", 50)
    reps = cfg.get("reps", 3)   
    conf_vals = cfg.get("conf_vals", [16,32])
    
    with Pool(processes=None) as p:
        p.map(main, conf_vals)
        
