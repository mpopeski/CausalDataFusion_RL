#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ast import literal_eval
import os

from strategies import naive_strategy, backdoor_strategy, frontdoor_strategy

class Vmax:
    
    def __init__(self, env, gamma, m, eta, Rmax, K):
        self.Rmax = Rmax
        self.Vmax = Rmax / (1-gamma)
        self.H = env.H
        self.K = K
        self.eta = eta
        self.gamma = gamma
        self.m = m
        self.env = env
        
        self.states = pd.Series(env.states).apply(str)
        self.actions = pd.Series(env.actions).apply(str)
        self.S = len(self.states)
        self.A = len(self.actions)
        
        self.SA = pd.MultiIndex.from_product([self.states, self.actions], names = ["s", "a"])
        self.SAS = pd.MultiIndex.from_product([self.states, self.actions, self.states], names = ["s", "a", "s'"])
                    
        self.SA_count = pd.DataFrame(0, index = self.SA, columns = ["count"])
        self.SAS_count = pd.DataFrame(0, index = self.SA, columns = self.states)
        self.SA_reward = pd.DataFrame(0, index = self.SA, columns = ["total"])
        self.R_sa = pd.DataFrame(0, index = self.SA, columns = ["reward"])
        
        self.data = pd.DataFrame(index = range(K), columns = range(self.H), dtype = object)
        
        self.Q = pd.DataFrame(self.Vmax, index = self.SA, columns = ["value"], dtype = float)
        
        self.reward = pd.Series(0, index = range(K*self.H), dtype = float)
        
    def update_counts(self, state, action, next_state, reward):
        self.SA_count.loc[state, action] += 1
        self.SAS_count[next_state].loc[state, action] += 1
        self.SA_reward.loc[state, action] += reward
        self.R_sa["reward"].loc[state, action] = self.SA_reward["total"].loc[state, action] / self.SA_count["count"].loc[state, action]
        
        
    def Q_update(self, state, action, next_state, reward, k):
        self.update_counts(state, action, next_state, reward)
        self.VI()
        
    def VI(self):
        mask = self.SA_count["count"] > 0
        P_SAS = self.SAS_count.loc[mask].divide(self.SA_count["count"].loc[mask], axis = 0)
        delta = 0
        first = 1
        while first or (delta > self.eta):
            q = self.Q.copy()["value"].loc[mask]
            delayed_value = self.gamma * P_SAS.multiply(self.Q.groupby(level = 0).max()["value"]).sum(axis = 1)
            imediate_reward = self.R_sa["reward"].loc[mask]
            Q = imediate_reward + delayed_value
            Q_ = (self.SA_count["count"].loc[mask] / self.m) * Q + \
                (1 - self.SA_count["count"].loc[mask] / self.m) * self.Vmax
            self.Q["value"].loc[mask] = Q_
            delta = (q - Q_).abs().max()
            first = 0            
    
    def policy(self, state):
        Qs = self.Q["value"].loc[str(state), :]
        action = Qs.idxmax()
        return literal_eval(action)
        
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
    
    def gen_init(self, N_sa, R_sa, P_sas):
        
        self.SA_count["count"] += N_sa.clip(lower = 0, upper = self.m)
        self.SA_reward["total"] += R_sa.multiply(self.SA_count["count"], axis = 0)
        self.SAS_count += P_sas.multiply(self.SA_count["count"], axis = 0)
        
        mask = self.SA_count["count"] > 0
        self.R_sa["reward"].loc[mask] = self.SA_reward["total"].loc[mask].divide(self.SA_count["count"].loc[mask], axis = 0)
        
        if mask.sum() > 0:
            self.VI()
    
    def initialize(self, how = 'ignore', size = None):
        
        if not size:
            size = len(self.env.data)
            
        if how == "ignore":
            print("Ignoring the observational samples")
            return 0
        elif how == "naive":
            print("Naively integrating the observational data")
            N_sa, R_sa, P_sas = naive_strategy(self.env.data.iloc[:size], self.env.S_index, self.env.SA_index)
        elif how == "controlled":
            print("Integrating the observational data with controlled confounding")
            N_sa, R_sa, P_sas = backdoor_strategy(self.env.data.iloc[:size], self.env.S_index, self.env.SU_index, self.env.SUA_index)
        elif how == "controlled_FD":
            print("Integrating the observational data with controlled confounding using frontdoor criterion")
            N_sa, R_sa, P_sas = frontdoor_strategy(self.env.data.iloc[:size], self.env.S_index, self.env.SA_index, self.env.SAM_index)
        else:
            raise ValueError("Not a valid initialization method. Choose from: ignore, naive, controlled")
        
        self.gen_init(N_sa, R_sa, P_sas)
        
        print(how, " - percentage:", self.SA_count.sum() / (len(self.states)*len(self.actions)*self.m))
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.Q.to_csv(path + "Q.csv")
        self.SA_count.to_csv(path + "SA_counts.csv")
        self.SAS_count.to_csv(path + "SAS_counts.csv")
        self.SA_reward.to_csv(path + "SA_reward.csv")
        return self.env.reward_states
                
                    