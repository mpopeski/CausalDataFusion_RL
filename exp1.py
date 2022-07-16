#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool

from Rmax import R_MAX
from Vmax import Vmax
from MDP_environments import TabularMDP

env = TabularMDP(5, 0, 500, default_prob = 4, n_reward_states = 12, policy = "v3_eng", simpson = True)
path = "../Final2/final_exp1/"

def main(K_obs):
    data = env.get_obs_data(K_obs)    
    m = 1000
    K_int = 300
    
    integration = ["ignore", "naive", "controlled"]
    integration_index = ["ignore_Rmax", "ignore_Vmax", "naive_Rmax", "naive_Vmax","controlled_Rmax", "controlled_Vmax"]
    #integration_index = ["ignore_Rmax", "naive_Rmax", "controlled_Rmax"]
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    
    path_ = path + f"{K_obs}_{K_int}/" 
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
    sizes = [500, 1000, 2500, 5000]
    #sizes = [100,200,300,400,500]
    print("starting to learn different models")
    with Pool(processes=None) as p:
        p.map(main, sizes)
