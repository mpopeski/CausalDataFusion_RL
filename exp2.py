#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool

from Rmax import R_MAX
from Vmax import Vmax
from MoRmax import MoRmax
from MDP_environments import TabularMDP


path = "../MoRmax_final2/final_exp2/"
load_path = "../Final10/final_exp2/"
K_obs = 5000 

def main(conf_val):
    path_ = path + f"{conf_val}/" 
    load_path_ = load_path + f"{conf_val}/Environment/"
    
    os.makedirs(path_, exist_ok=True)

    env = TabularMDP(state_values = 5, action_values = 3, H = 500, default_prob = 4, n_reward_states=12, policy = "random", 
                     simpson = False, conf_values = conf_val)
    
    if load_path_:
        env.load_env(load_path_)
    else:
        env.save_env(path_ + "Environment/")
    
    print("collecting observational data")
    data = env.get_obs_data(K_obs)   
    m = 1000
    K_int = 500
    
    integration = ["ignore", "controlled", "controlled_FD"]
    #integration_index = ["ignore_Rmax", "ignore_Vmax", "controlled_Rmax",\
    #                     "controlled_Vmax", "controlled_FD_Rmax", "controlled_FD_Vmax"]
    integration_index = ["ignore_MoRmax", "controlled_MoRmax", "controlled_FD_MoRmax"]
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    
    for rep in range(reps):
        results = []
        for integ in integration:
            """
            model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
            model1.initialize(data, integ)
            model1.learn()
            results.append(model1.reward)
            
            model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
            model2.initialize(data, integ)
            model2.learn()
            results.append(model2.reward)
            """
            model3 = MoRmax(env, gamma, m, eta, Rmax, K_int)
            model3.initialize(data, integ)
            model3.learn()
            results.append(model3.reward)
            
        results = pd.DataFrame(results, index = integration_index).T
        results.to_csv(path_ + f"results{rep}.csv")
          
    data["r"].to_csv(path_ + "obs_rew.csv")

if __name__ == "__main__":

    conf_vals = [4,8,16,32,64]

    with Pool(processes=None) as p:
        p.map(main, conf_vals)
        
