#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from Rmax import R_MAX
from MDP_environments import TabularMDP
import os
from multiprocessing import Pool

def main(config):
    K_obs = config[0]
    K_int = config[1]
    m = config[2]
    path = config[3]
    path_ = path + f"Rmax/{K_obs}_{K_int}_{m}/"
    os.makedirs(path_, exist_ok=True)
    env = TabularMDP(5, 0, 80, n_reward_states = 12, policy = "v3_eng", simpson = True)
    
    #path_environment = path_ + "environment/"
    #env.save_env(path_environment)
    
    env.observational_data(K_obs)
    integration = ["ignore", "naive", "controlled"]
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 5
    for rep in range(reps):
        results = []
        for integ in integration:
            model = R_MAX(env, gamma, m, eta, Rmax, K_int)
            model.initialize(integ)
            model.learn()
            results.append(model.reward.cumsum())
            print(model.save_model(path_ + f"{integ}/{rep}/"))
    
        results = pd.DataFrame(results, index = integration).T
        results.to_csv(path_ + f"results{rep}.csv")

if __name__ == "__main__":
    path = "../experiments_v22/"
    configs =[(1000, 2000, 1000, path), (3000, 2000, 1000, path),\
               (5000, 2000, 1000, path)]
    with Pool(processes=None) as p:
        p.map(main, configs)
