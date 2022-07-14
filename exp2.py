#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool

from Rmax import R_MAX
from Vmax import Vmax
from MDP_environments import TabularMDP


def main(config):
    path = config[0]
    conf_val = config[1]
    m = 1000
    K_int = 500
    K_obs = 5000
    # the environment
    env = TabularMDP(state_values = 5, action_values = 2, H = 800, n_reward_states=12, policy = "random", 
                     simpson = True, conf_values = conf_val)
    print("collecting observational data")
    env.observational_data(K_obs)   
    
    integration = ["ignore", "controlled", "controlled_FD"]
    integration_index = ["ignore_Rmax", "ignore_Vmax", "controlled_Rmax",\
                         "controlled_Vmax", "controlled_FD_Rmax", "controlled_FD_Vmax"]
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    
    path_ = path + f"{conf_val}_{K_int}/" 
    os.makedirs(path_, exist_ok=True)
        
    for rep in range(reps):
        results = []
        for integ in integration:
            model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
            model1.initialize(integ)
            model1.learn()
            results.append(model1.reward)
            
            model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
            model2.initialize(integ)
            model2.learn()
            results.append(model2.reward)
            
        results = pd.DataFrame(results, index = integration_index).T
        results.to_csv(path_ + f"results{rep}.csv")
    
    env.data["r"].to_csv(path_ + "obs_rew.csv")

if __name__ == "__main__":
    path = "../final_exp2_repeat/"
    
    configs = [(path, 4), (path, 8), (path, 16)]

    with Pool(processes=None) as p:
        p.map(main, configs)
