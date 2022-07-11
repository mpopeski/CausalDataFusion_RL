#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool

from Rmax import R_MAX
from Vmax import Vmax
from MDP_environments import TabularMDP

K_obs = 50000
# the environment
env = TabularMDP(state_values = 5, action_values = 0, H = 80, n_reward_states=12, policy = "v3_eng", simpson = True)
print("collecting observational data")
env.observational_data(K_obs)
    
def main(config):
    path = config[0]
    size = config[1]
    
    # number that I am gonna modify later
    m = 1000
    K_int = 5000
    
    integration = ["naive", "controlled"]
    integration_index = ["naive_Rmax", "naive_Vmax","controlled_Rmax", "controlled_Vmax"]
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    
    path_ = path + f"{size}_{K_int}/" 
    os.makedirs(path_, exist_ok=True)
        
    for rep in range(reps):
        results = []
        for integ in integration:
            model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
            model1.initialize(integ, size)
            model1.learn()
            results.append(model1.reward.cumsum())
            
            model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
            model2.initialize(integ, size)
            model2.learn()
            results.append(model2.reward.cumsum())
            
        results = pd.DataFrame(results, index = integration_index).T
        results.to_csv(path_ + f"results{rep}.csv")

def main2(path):
    
    m = 1000
    K_int = 5000
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    
    print("only online learning")
    for rep in range(reps):
        results = []
        model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
        model1.initialize("ignore")
        model1.learn()
        results.append(model1.reward.cumsum())
        
        model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
        model2.initialize("ignore")
        model2.learn()
        results.append(model2.reward.cumsum())
        
        results = pd.DataFrame(results, index = ["ignore_Rmax", "ignore_Vmax"]).T
        results.to_csv(path + f"results{rep}.csv")
        

if __name__ == "__main__":
    path = "../final_exp1/"
    
    sizes = [10000, 20000, 30000, 40000, 50000]#, 60000, 70000, 80000, 90000, 100000]
    #sizes = [100,200,300,400,500]
    configs = []
    for size in sizes:
        configs.append((path, size))
    
    with Pool(processes=None) as p:
        p.map(main, configs)

    main2(path)
    env.data["r"].to_csv(path + "obs_rew.csv")