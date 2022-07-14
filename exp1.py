#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from multiprocessing import Pool
import copy

from Rmax import R_MAX
from Vmax import Vmax
from MDP_environments import TabularMDP

K_obs = 5000
# the environment
base_env = TabularMDP(state_values = 5, action_values = 0, H = 800, n_reward_states=12, policy = "v3_eng", simpson = True)
print("collecting observational data")
base_env.observational_data(K_obs)
    
def main(config):
    path = config[0]
    size = config[1]
    env = copy.deepcopy(base_env)
    # number that I am gonna modify later
    m = 1000
    K_int = 500
    
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
            results.append(model1.reward)
            
            model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
            model2.initialize(integ, size)
            model2.learn()
            results.append(model2.reward)
            
        results = pd.DataFrame(results, index = integration_index).T
        results.to_csv(path_ + f"results{rep}.csv")

def main2(path):
    
    m = 1000
    K_int = 500
    
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    env = copy.deepcopy(base_env)
    print("only online learning")
    for rep in range(reps):
        results = []
        model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
        model1.initialize("ignore")
        model1.learn()
        results.append(model1.reward)
        
        model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
        model2.initialize("ignore")
        model2.learn()
        results.append(model2.reward)
        
        results = pd.DataFrame(results, index = ["ignore_Rmax", "ignore_Vmax"]).T
        results.to_csv(path + f"results{rep}.csv")
        

if __name__ == "__main__":
    path = "../final_exp1/"
    
    sizes = [1000, 2000, 3000, 4000, 5000]#, 60000, 70000, 80000, 90000, 100000]
    #sizes = [100,200,300,400,500]
    configs = []
    for size in sizes:
        configs.append((path, size))
    
    with Pool(processes=len(sizes)) as p:
        p.map(main, configs)

    main2(path)
    env.data["r"].to_csv(path + "obs_rew.csv")