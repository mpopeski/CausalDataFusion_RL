#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from Vmax import Vmax
from MDP_environments import TabularMDP
import os
from multiprocessing import Pool

def main(config):
    K_obs = config[0]
    K_int = config[1]
    m = config[2]
    path = config[3]
    path_ = path + f"Vmax/{K_obs}_{K_int}_{m}/"
    os.makedirs(path_, exist_ok=True)
    env = TabularMDP(2, 5, 2, [-1,0,1], 10, n_reward_states = 12, policy = "v2_eng")
    
    path_environment = path_ + "environment/"
    env.save_env(path_environment)
    
    env.observational_data(K_obs)
    integration = ["ignore", "naive", "controlled"]
    gamma = 0.9
    eta = 0.0001
    Rmax = 1
    reps = 3
    for rep in range(reps):
        results = []
        for integ in integration:
        
            model = Vmax(env, gamma, m, eta, Rmax, K_int)
            model.initialize(integ)
            model.learn()
            results.append(model.reward.cumsum())
            print(model.save_model(path_ + f"{integ}/{rep}/"))
    
        results = pd.DataFrame(results, index = integration).T
        results.to_csv(path_ + f"results{rep}.csv")

if __name__ == "__main__":
    path = "../experiments_v18/"
    configs = [(25000, 20000, 1000, path), (29000, 20000, 1000, path), (33000, 20000, 1000, path),
               (26000, 20000, 1000, path), (30000, 20000, 1000, path), (34000, 20000, 1000, path),
               (27000, 20000, 1000, path), (31000, 20000, 1000, path), (35000, 20000, 1000, path),
               (28000, 20000, 1000, path), (32000, 20000, 1000, path), (36000, 20000, 1000, path)]
    with Pool(processes=None) as p:
        p.map(main, configs)

