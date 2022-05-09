#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from Vmax import Vmax
from MDP_environments import TabularMDP
import os

def main(K_obs, K_int, m, path):
    path_ = path + f"Vmax/{K_obs}_{K_int}_{m}/"
    os.makedirs(path_, exist_ok=True)
    env = TabularMDP(2, 5, 2, [-1,0,1], 8)
    env.observational_data(K_obs)
    integration = ["ignore", "naive", "controlled"]
    gamma = 0.9
    eta = 0.0001
    Rmax = 0.2
    reps = 10
    for rep in range(reps):
        results = []
        for integ in integration:
        
            model = Vmax(env, gamma, m, eta, Rmax, K_int)
            model.initialize(integ)
            model.learn()
            results.append(model.reward.cumsum())
            print(model.save_model(path_ + f"{integ}/{rep}/"))
    
        results = pd.DataFrame(results, index = integration)
        results.to_csv(path_ + f"results{rep}.csv")

if __name__ == "__main__":
    configs = [(5000,200,500), (10000, 500, 1000) ,(10000, 1000, 1000), (15000, 500, 1000),
               (5000,5000,500), (5000, 2500, 500), (5000,5000,1000)]
    path = "../experiments/"
    for config in configs:
        main(config[0], config[1], config[2], path)
