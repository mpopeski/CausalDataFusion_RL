#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ast import literal_eval
import os

from strategies import naive_strategy, backdoor_strategy, frontdoor_strategy

class R_MAX:
    
    def __init__(self, env, gamma, m, eta, Rmax, K):
        
        # General Rmax parameters 
        self.Rmax = Rmax
        self.Vmax = Rmax / (1-gamma)
        self.H = env.H
        self.K = K
        self.eta = eta
        self.gamma = gamma
        self.m = m
        
        # the environment 
        self.env = env
        
        # seting indices 
        self.states = env.S_index
        self.actions = pd.Series(env.actions).apply(str)
        self.SA = env.SA_index
        
        # set and action space cardinality
        self.S = len(self.states)
        self.A = len(self.actions)
        
        # initializing Rmax            
        self.SA_count = pd.DataFrame(0, index = self.SA, columns = ["count"])
        self.SAS_count = pd.DataFrame(0, index = self.SA, columns = self.states)
        self.SA_reward = pd.DataFrame(0, index = self.SA, columns = ["total"])
        self.R_sa = pd.DataFrame(self.Rmax, index = self.SA, columns = ["reward"])
        self.Q = pd.DataFrame(self.Vmax, index = self.SA, columns = ["value"], dtype = float)
        self.P_sas = pd.DataFrame(0, index = self.SA, columns = self.states)
        
        for state in self.states:
            self.P_sas[state].loc[state, :] = 1
        
        # Initializing history counters for the whole data and only the reward
        self.data = pd.DataFrame(index = range(K), columns = range(self.H), dtype = object)
        self.reward = pd.Series(0, index = range(K*self.H), dtype = float)
    
    # Rmax Learning loop with Value iteration
    def learn(self):
        for k in range(self.K):
            state = self.env.start()
            for h in range(self.H):
                action = self.policy(state)
                m, reward, next_state = self.env.transition(state, action)
                if self.SA_count["count"].loc[str(state), str(action)] < self.m:
                    self.Q_update(str(state), str(action), str(next_state), reward, k*self.H + h)
                self.reward[k*self.H + h] = reward
                state = next_state
    
    def policy(self, state):
        Qs = self.Q["value"].loc[str(state), :]
        action = Qs.sample(frac = 1.).idxmax()
        return literal_eval(action)
    
    def Q_update(self, state, action, next_state, reward, k):
        self.update_counts(state, action, next_state, reward, k)
        if self.SA_count["count"].loc[state, action] >= self.m:
            self.R_sa["reward"].loc[state, action] = self.SA_reward["total"].loc[state, action] / self.m
            self.P_sas.loc[state, action] = self.SAS_count.loc[state, action].divide(self.SA_count["count"].loc[state, action], axis = 0)
            self.VI()
            
    def update_counts(self, state, action, next_state, reward, k):
        self.SA_count.loc[state, action] += 1
        self.SAS_count[next_state].loc[state, action] += 1
        self.SA_reward.loc[state, action] += reward
            
    def VI(self):
        delta = 0
        first = 1
        while first or (delta > self.eta):
            q = self.Q.copy()["value"]
            delayed_value = self.gamma * self.P_sas.multiply(self.Q.groupby(level = 0).max()["value"]).sum(axis = 1)
            imediate_reward = self.R_sa["reward"]
            Q = imediate_reward + delayed_value
            self.Q["value"] = Q
            delta = (q - Q).abs().max()
            first = 0            
    
    # Initialization from observational data based on different strategies
    def initialize(self, data, how = 'ignore'):

        if how == "ignore":
            print("Ignoring the observational samples")
            return 0
        elif how == "naive":
            print("Naively integrating the observational data")
            N_sa, R_sa, P_sas = naive_strategy(data, self.env.S_index, self.env.SA_index)
        elif how == "controlled":
            print("Integrating the observational data with controlled confounding")
            N_sa, R_sa, P_sas = backdoor_strategy(data, self.env.S_index, self.env.SU_index, self.env.SUA_index)
        elif how == "controlled_FD":
            print("Integrating the observational data with controlled confounding using frontdoor criterion")
            N_sa, R_sa, P_sas = frontdoor_strategy(data, self.env.S_index, self.env.SA_index, self.env.SAM_index)
        else:
            raise ValueError("Not a valid initialization method. Choose from: ignore, naive, controlled")
        
        self.gen_init(N_sa, R_sa, P_sas)
        
    # Rmax initialization from observational estimates
    def gen_init(self, N_sa, R_sa, P_sas):
        
        self.SA_count["count"] += N_sa.clip(lower = 0, upper = self.m)
        self.SA_reward["total"] += R_sa.multiply(self.SA_count["count"], axis = 0)
        self.SAS_count += P_sas.multiply(self.SA_count["count"], axis = 0)
        
        mask = self.SA_count["count"] >= self.m
        self.R_sa["reward"].loc[mask] = self.SA_reward["total"].loc[mask] / self.m
        self.P_sas.loc[mask] = self.SAS_count.loc[mask] / self.m
        if mask.sum() > 0:
            self.VI()
        
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.Q.to_csv(path + "Q.csv")
        self.SA_count.to_csv(path + "SA_counts.csv")
        self.SAS_count.to_csv(path + "SAS_counts.csv")
        self.SA_reward.to_csv(path + "SA_reward.csv")
        return self.env.reward_states
                