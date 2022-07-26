#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ast import literal_eval
import os

from strategies import naive_strategy, backdoor_strategy, frontdoor_strategy

class MoRmax:
    
    def __init__(self, env, gamma, m, eta, Rmax, K):
        self.Rmax = Rmax
        self.Vmax = Rmax / (1-gamma)
        self.H = env.H
        self.K = K
        self.eta = eta
        self.gamma = gamma
        self.m = m
        self.eta2 = ((1-gamma)*eta)/15
        self.t = 0
        self.tao = 0
        
        self.env = env
        
        
        self.states = pd.Series(env.states).apply(str)
        self.actions = pd.Series(env.actions).apply(str)
        self.S = len(self.states)
        self.A = len(self.actions)
        
        self.SA = pd.MultiIndex.from_product([self.states, self.actions], names = ["s", "a"])
        self.SAS = pd.MultiIndex.from_product([self.states, self.actions, self.states], names = ["s", "a", "s'"])
        
        self.update_time = pd.Series(0, index = self.SA)
                    
        self.SA_count = pd.DataFrame(0, index = self.SA, columns = ["count"])
        self.SAS_count = pd.DataFrame(0, index = self.SA, columns = self.states)
        self.SA_reward = pd.DataFrame(0, index = self.SA, columns = ["total"])
        self.R_sa = pd.DataFrame(self.Rmax, index = self.SA, columns = ["reward"])
        self.P_sas = pd.DataFrame(0, index = self.SA, columns = self.states)
        self.Q = pd.DataFrame(self.Vmax, index = self.SA, columns = ["value"], dtype = float)
        self.Q_prev = pd.DataFrame(self.Vmax, index = self.SA, columns = ["value"], dtype = float)
        
        
        for state in self.states:
            self.P_sas[state].loc[state, :] = 1
        
        self.data = pd.DataFrame(index = range(K), columns = range(self.H), dtype = object)
        
        self.reward = pd.Series(0, index = range(K*self.H), dtype = float)
        
    def update_counts(self, state, action, next_state, reward, k):
        self.SA_count.loc[state, action] += 1
        self.SAS_count[next_state].loc[state, action] += 1
        self.SA_reward.loc[state, action] += reward
        self.t += 1
        
    def Q_update(self, state, action, next_state, reward, k):
        self.update_counts(state, action, next_state, reward, k)
        if self.SA_count["count"].loc[state, action] >= self.m:
            cond1 = self.update_time.loc[state, action] == 0
            cond2 = (self.tao > self.update_time.loc[state, action])  and\
                (self.Q_prev["value"].loc[state,action] - self.Q["value"].loc[state,action] > self.eta2)
                
            if cond1 or cond2:
                P_sas = self.P_sas.copy()
                R_sa = self.R_sa.copy()
                R_sa["reward"].loc[state, action] = self.SA_reward["total"].loc[state, action] / self.m
                P_sas.loc[state, action] = self.SAS_count.loc[state, action].divide(self.SA_count["count"].loc[state, action], axis = 0)
                Q_ = self.VI(P_sas, R_sa)
                if Q_.loc[state, action] <= self.Q["value"].loc[state, action]:
                    self.P_sas = P_sas
                    self.R_sa = R_sa
                    self.Q["value"] = Q_
            
                self.update_time.loc[state, action] = self.t
                self.tao = self.t
                self.Q_prev["value"].loc[state, action] = self.Q["value"].loc[state, action]
            
            #this is outside the if condition
            self.SA_count["count"].loc[state, action] = 0
            self.SAS_count.loc[state, action] = 0
            self.SA_reward["total"].loc[state, action] = 0
            
        
    def VI(self, P_sas, R_sa):
        delta = 0
        first = 1
        Q = self.Q.copy()["value"]
        while first or (delta > self.eta):
            q = Q.copy()
            delayed_value = self.gamma * P_sas.multiply(Q.groupby(level = 0).max()).sum(axis = 1)
            imediate_reward = R_sa["reward"]
            Q = imediate_reward + delayed_value
            delta = (q - Q).abs().max()
            first = 0
        return Q
    
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
                # always do the update
                self.Q_update(str(state), str(action), str(next_state), reward, k*self.H + h)
                self.reward[k*self.H + h] = reward
                state = next_state
    
    def gen_init(self, N_sa, R_sa, P_sas):
        
        self.SA_count["count"] += N_sa.clip(lower = 0, upper = self.m)
        self.SA_reward["total"] += R_sa.multiply(self.SA_count["count"], axis = 0)
        self.SAS_count += P_sas.multiply(self.SA_count["count"], axis = 0)
        
        mask = self.SA_count["count"] >= self.m
        if mask.sum() > 0:
            R_sa = self.R_sa.copy()
            P_sas = self.P_sas.copy()
            R_sa["reward"].loc[mask] = self.SA_reward["total"].loc[mask] / self.m
            P_sas.loc[mask] = self.SAS_count.loc[mask] / self.m
            Q_ = self.VI(P_sas, R_sa)
            if np.any(Q_<= self.Q["value"]):
                self.P_sas = P_sas
                self.R_sa = R_sa
                self.Q["value"] = Q_
            
            self.Q_prev["value"].loc[mask] = self.Q["value"].loc[mask]
            self.SA_count["count"].loc[mask] = 0
            self.SAS_count.loc[mask] = 0
            self.SA_reward["total"].loc[mask] = 0
                     
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
        
        print(how, " - percentage:", self.SA_count.sum() / (len(self.states)*len(self.actions)*self.m))
        
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.Q.to_csv(path + "Q.csv")
        self.SA_count.to_csv(path + "SA_counts.csv")
        self.SAS_count.to_csv(path + "SAS_counts.csv")
        self.SA_reward.to_csv(path + "SA_reward.csv")
        return self.env.reward_states
                