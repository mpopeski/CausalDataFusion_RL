#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ast import literal_eval
import os


class R_MAX:
    
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
        self.R_sa = pd.DataFrame(self.Rmax, index = self.SA, columns = ["reward"])
        self.Q = pd.DataFrame(self.Vmax, index = self.SA, columns = ["value"], dtype = float)
        self.P_sas = pd.DataFrame(0, index = self.SA, columns = self.states)
        
        for state in self.states:
            self.P_sas[state].loc[state, :] = 1
        
        self.data = pd.DataFrame(index = range(K), columns = range(self.H), dtype = object)
        
        self.reward = pd.Series(0, index = range(K*self.H), dtype = float)
        
    def update_counts(self, state, action, next_state, reward, k):
        self.SA_count.loc[state, action] += 1
        self.SAS_count[next_state].loc[state, action] += 1
        self.SA_reward.loc[state, action] += reward
        # self.R_sa["reward"].loc[state, action] = self.SA_reward["total"].loc[state, action] / self.SA_count["count"].loc[state, action]
        
        
    def Q_update(self, state, action, next_state, reward, k):
        self.update_counts(state, action, next_state, reward, k)
        if self.SA_count["count"].loc[state, action] >= self.m:
            self.R_sa["reward"].loc[state, action] = self.SA_reward["total"].loc[state, action] / self.m
            self.P_sas.loc[state, action] = self.SAS_count.loc[state, action].divide(self.SA_count["count"].loc[state, action], axis = 0)
            self.VI()
        
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
    
    def policy(self, state):
        Qs = self.Q["value"].loc[str(state), :]
        action = Qs.sample(frac = 1.).idxmax()
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
    
    def initialize(self, how = 'ignore'):
        if len(self.env.data):
            if how == "ignore":
                print("Ignoring the observational data")
                
            elif how == "naive":
                print("Naively integrating the observational data")
                SA_count = self.env.counts["SA"]
                SAS_count = self.env.counts["SAS"]
                SA_reward = self.env.counts["SA_reward"]
                P_SAS = SAS_count.divide(SA_count["count"], axis = 0).fillna(0)
                self.SA_count += SA_count.clip(lower = 0, upper = self.m)
                self.SAS_count += P_SAS.multiply(self.SA_count["count"], axis = 0, level = 0) 
                self.SA_reward["total"] += SA_reward["total"].divide(SA_count["count"].clip(lower = 1, upper = None), axis = 0) * \
                    SA_count.clip(lower = 0, upper = self.m)["count"]
                
                mask = self.SA_count["count"] >= self.m
                self.R_sa["reward"].loc[mask] = self.SA_reward["total"].loc[mask] / self.m
                self.P_sas.loc[mask] = self.SAS_count.loc[mask] / self.m
                if mask.sum() > 0:
                    self.VI()
                    
            elif how == "controlled":
                print("Integrating the observational data with controlled confounding")
                P_us = (self.env.counts["SU"].divide(self.env.counts["S"]["count"].clip(lower = 1, upper = None), axis=0))
                P_suas1 = self.env.counts["SUAS"][0].divide(self.env.counts["SUA"][0]["count"].clip(lower = 1, upper = None), axis = 0)
                P_suas2 = self.env.counts["SUAS"][1].divide(self.env.counts["SUA"][1]["count"].clip(lower = 1, upper = None), axis = 0)
                P_sas = P_suas1.multiply(P_us[0], axis = 0, level = 0) + P_suas2.multiply(P_us[1], axis = 0, level = 0)
                
                
                SUA_reward =  self.env.counts["SUA_reward"]
                R_sua0 = SUA_reward[0]["total"].divide(self.env.counts["SUA"][0]["count"].clip(lower = 1, upper = None), axis = 0) 
                R_sua1 = SUA_reward[1]["total"].divide(self.env.counts["SUA"][1]["count"].clip(lower = 1, upper = None), axis = 0)
                R_sa = R_sua0.multiply(P_us[0], axis = 0, level = 0) + R_sua1.multiply(P_us[1], axis = 0, level = 0)
                #SA_reward = R_sa.multiply(SA_count["count"].clip(lower = 0, upper = self.m), axis = 0, level = 0)
                
                SA_count = self.env.counts["SUA"][0]["count"].combine(self.env.counts["SUA"][1]["count"], min)
                SAS_count = P_sas.multiply(SA_count.clip(lower = 0, upper = self.m), axis = 0, level = 0)
                self.SA_count["count"] += SAS_count.sum(axis = 1)
                self.SAS_count += SAS_count
                self.SA_reward["total"] += R_sa.multiply(self.SA_count["count"], axis = 0, level = 0)
                
                mask = self.SA_count["count"] >= self.m
                self.R_sa["reward"].loc[mask] = self.SA_reward["total"].loc[mask] / self.m
                self.P_sas.loc[mask] = self.SAS_count.loc[mask] / self.m
                if mask.sum() > 0:
                    self.VI()
                
            elif how == "controlled_FD":
                print("Integrating the Observational data with controlled confounding using frontdoor criterion")
                S_count = self.env.counts["S"]
                SA_count = self.env.counts["SA"]
                SAM_count = self.env.counts["SAM"]
                SAMS_count = self.env.counts["SAMS"]
                SAM_reward = self.env.counts["SAM_reward"]
                mediators = self.env.mediators
                P_sa = SA_count.divide(S_count["count"], axis = 0, level = 0)
                P_sas = pd.DataFrame(0, index = self.env.counts["SAS"].index , columns = self.env.counts["SAS"].columns)
                R_sa = pd.Series(0, index = SA_count.index)
                for m in mediators:
                    P_sam = SAM_count[m].divide(SA_count["count"].clip(lower = 1, upper = None), axis = 0)
                    P_sams = SAMS_count[m].divide(SAM_count[m]["count"].clip(lower = 1, upper = None), axis = 0)
                    inner_control = P_sams.multiply(P_sa["count"], axis = 0).groupby(level = 0).sum()
                    P_sas += inner_control.multiply(P_sam["count"], axis = 0, level = 0)
                    R_sam = SAM_reward[m].divide(SAM_count[m]["count"].clip(lower = 1, upper = None), axis = 0)
                    inner_control_Rs = R_sam.multiply(P_sa["count"], axis = 0).groupby(level = 0).sum()
                    R_sa += inner_control_Rs.multiply(P_sam["count"], axis = 0, level =0)["count"]
                
                SAS_count = P_sas.multiply(SA_count["count"].clip(lower = 0, upper = self.m), axis = 0, level = 0)
                self.SA_count["count"] += SAS_count.sum(axis = 1)
                self.SAS_count += SAS_count
                self.SA_reward["total"] += R_sa.multiply(self.SA_count["count"], axis = 0)
                
                mask = self.SA_count["count"] >= self.m
                self.R_sa["reward"].loc[mask] = self.SA_reward["total"].loc[mask] / self.m
                self.P_sas.loc[mask] = self.SAS_count.loc[mask] / self.m
                if mask.sum() > 0:
                    self.VI()
                
            else:
                raise ValueError("Not a valid initialization method. Choose from: ignore, naive, controlled")
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.Q.to_csv(path + "Q.csv")
        self.SA_count.to_csv(path + "SA_counts.csv")
        self.SAS_count.to_csv(path + "SAS_counts.csv")
        self.SA_reward.to_csv(path + "SA_reward.csv")
        return self.env.reward_states
                
            
        