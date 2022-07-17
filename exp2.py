#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool
import copy

from Rmax import R_MAX
from Vmax import Vmax
from MDP_environments import TabularMDP


path = "../Final3/final_exp2/"
K_obs = 5000 

def main(conf_val):

    env = TabularMDP(state_values = 5, action_values = 3, H = 500, default_prob = 4, n_reward_states=12, policy = "random", 
                     simpson = False, conf_values = conf_val)
    
    print("collecting observational data")
    data = env.get_obs_data(K_obs)   
    m = 1000
    K_int = 400
    
    integration = ["ignore", "controlled", "controlled_FD"]
    integration_index = ["ignore_Rmax", "ignore_Vmax", "controlled_Rmax",\
                         "controlled_Vmax", "controlled_FD_Rmax", "controlled_FD_Vmax"]
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    
    path_ = path + f"{conf_val}/" 
    os.makedirs(path_, exist_ok=True)
    
    for rep in range(reps):
        results = []
        for integ in integration:
            model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
            model1.initialize(data, integ)
            model1.learn()
            results.append(model1.reward)
            
            model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
            model2.initialize(data, integ)
            model2.learn()
            results.append(model2.reward)
            
        results = pd.DataFrame(results, index = integration_index).T
        results.to_csv(path_ + f"results{rep}.csv")
          
    data["r"].to_csv(path_ + "obs_rew.csv")

if __name__ == "__main__":

    conf_vals = [4,8,16,32,64]

    with Pool(processes=None) as p:
        p.map(main, conf_vals)
        
