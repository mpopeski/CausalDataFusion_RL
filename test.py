#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from UCBVI import UCBVI
from MDP_environments import TabularMDP
import os

def main(K_obs, K_int, path):
    path_ = path + f"{K_obs}_{K_int}/"
    os.makedirs(path_, exist_ok=True)
    env = TabularMDP(2, 3, 2, [-1,0,1], 5)
    env.observational_data(K_obs)
    integration = ["ignore", "naive", "controlled"]
    delta = 0.3
    results = []
    for integ in integration:
        model = UCBVI(env, K_int, delta)
        model.initialize(integ)
        model.learn()
        results.append(model.cumreward.cumsum())
        model.V.to_csv(path_ + f"V_{integ}.csv")
        print(model.save_model(path_ + f"{integ}/"))
    
    results = pd.DataFrame(results, index = integration)
    results.to_csv(path_ + f"results.csv")

if __name__ == "__main__":
    Ks = [2000000]
    Kints = [200000]
    path = "../experiments/"
    for K_obs in Ks:
        for K_int in Kints:
            main(K_obs, K_int, path)
    
        
