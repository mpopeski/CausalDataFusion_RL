#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ast import literal_eval
import os


class Vmax:
    
    def __init__(self, env, gamma, m, eta, Rmax, K):
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
        
        self.cumreward = pd.Series(index = range(K*self.H), dtype = float)
        
    def update_counts(self, state, action, next_state, reward, k):
        self.SA_count.loc[state, action] += 1
        self.SAS_count[next_state].loc[state, action] += 1
        self.SA_reward.loc[state, action] += reward
        # self.data.loc[k, :] = pd.Series(episode, dtype = object)
        self.R_sa["reward"].loc[state, action] = self.SA_reward["total"].loc[state, action] / self.SA_count["count"].loc[state, action]
        #self.cumreward[k] = reward 
    
    def Q_update(self, state, action, next_state, reward, k):
        self.update_counts(state, action, next_state, reward, k)
        mask = self.SA_count["count"] > 0
        P_SAS = self.SAS_count.loc[mask].divide(self.SA_count["count"].loc[mask], axis = 0)
        self.VI(P_SAS, mask)
        
    def VI(self, P_SAS, mask):
        delta = 0
        first = 1
        while first or (delta > self.eta):
            q = self.Q.copy()["value"].loc[mask]
            delayed_value = self.gamma * P_SAS.multiply(self.Q.groupby(level = 0).max()["value"] + 1e-8).sum(axis = 1)
            imediate_reward = self.R_sa["reward"].loc[mask]
            Q = imediate_reward + delayed_value
            Q_ = (self.SA_count["count"].loc[mask] / self.m) * Q + \
                (1 - self.SA_count["count"].loc[mask] / self.m) * self.Vmax
            self.Q["value"].loc[mask] = Q_
            delta = max(delta, (q - Q_).abs().max())
            first = 0            
    
    def policy(self, state):
        Qs = self.Q["value"].loc[str(state), :]
        action = Qs.idxmax()[-1]
        return literal_eval(action)
        
    def learn(self):
        for k in range(self.K):
            state = self.env.start()
            for h in range(self.H):
                action = self.policy(state)
                reward, next_state = self.env.transition(state, action)
                if self.SA_count["count"].loc[str(state), str(action)] < self.m:
                    self.Q_update(str(state), str(action), str(next_state), reward, k*self.H + h)
                state = next_state
    
    def initialize(self, how = 'ignore'):
        if len(self.env.data):
            if how == "ignore":
                print("Ignoring the observational data")
            elif how == "naive":
                print("Naively integrating the observational data")
                self.SA_count += self.env.counts["SA"]
                self.SAS_count += self.env.counts["SAS"]
                self.SA_reward += self.env.counts["SA_reward"]
            elif how == "controlled":
                print("Integrating the observational data with controlled confounding")
                P_us = (self.env.counts["SU"].divide(self.env.counts["S"]["count"],axis=0)).fillna(0)
                P_suas1 = self.env.counts["SUAS"][0].divide(self.env.counts["SUA"][0]["count"], axis = 0).fillna(0)
                P_suas2 = self.env.counts["SUAS"][1].divide(self.env.counts["SUA"][1]["count"], axis = 0).fillna(0)
                P_sas = P_suas1.multiply(P_us[0], axis = 0, level = 0) + P_suas2.multiply(P_us[1], axis = 0, level = 0)
                SAS_count = P_sas.multiply(self.env.counts["SA"]["count"], axis = 0, level = 0)
                self.SA_count += self.env.counts["SA"]
                self.SA_reward += self.env.counts["SA_reward"]
                self.SAS_count += SAS_count
            else:
                raise ValueError("Not a valid initialization method. Choose from: ignore, naive, controlled")
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.Q.to_csv(path + "Q.csv")
        self.SA_count.to_csv(path + "SA_counts.csv")
        self.SAS_count.to_csv(path + "SAS_counts.csv")
        self.SA_reward.to_csv(path + "SA_reward.csv")
        return self.env.reward_states
                
            
        