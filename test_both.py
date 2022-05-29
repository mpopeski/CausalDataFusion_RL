#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from Rmax import R_MAX
from Vmax import Vmax
from MDP_environments import TabularMDP
import os
from multiprocessing import Pool

def main(config):
    K_obs = config[0]
    K_int = config[1]
    m = config[2]
    path = config[3]
    path_ = path + f"Both/{K_obs}_{K_int}_{m}/"
    os.makedirs(path_, exist_ok=True)
    env = TabularMDP(5, 6, 80, n_reward_states = 12, default_prob = 10, policy = "v3_eng", simpson = True)
    
    #path_environment = path_ + "environment/"
    #env.save_env(path_environment)
    
    env.observational_data(K_obs)
    integration = ["ignore", "naive", "controlled", "controlled_FD"]
    integration_index = ["ignore_Rmax", "ignore_Vmax", "naive_Rmax", "naive_Vmax",\
                         "controlled_Rmax", "controlled_Vmax", "controlled_FD_Rmax", "controlled_FD_Vmax"]
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    for rep in range(reps):
        results = []
        for integ in integration:
            model1 = R_MAX(env, gamma, m, eta, Rmax, K_int)
            model1.initialize(integ)
            model1.learn()
            results.append(model1.reward.cumsum())
            print(model1.save_model(path_ + f"{integ}/{rep}/Rmax/"))
            model2 = Vmax(env, gamma, m, eta, Rmax, K_int)
            model2.initialize(integ)
            model2.learn()
            results.append(model2.reward.cumsum())
            print(model2.save_model(path_ + f"{integ}/{rep}/Vmax/"))
        
        results = pd.DataFrame(results, index = integration_index).T
        results.to_csv(path_ + f"results{rep}.csv")

if __name__ == "__main__":
    path = "../experiments_v24/"
    configs =[(5000, 5000, 1000, path), (10000, 5000, 1000, path),
              (20000, 5000, 1000, path)]
    with Pool(processes=None) as p:
        p.map(main, configs)