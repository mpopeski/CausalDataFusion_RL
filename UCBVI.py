#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ast import literal_eval

class UCBVI:
    
    def __init__(self, env, K, delta):
        
        self.H = env.H
        self.K = K
        self.delta = delta
        self.env = env
        
        self.states = pd.Series(env.states).apply(str)
        self.actions = pd.Series(env.actions).apply(str)
        self.S = len(self.states)
        self.A = len(self.actions)
        self.L = np.log((5*self.S*self.A*self.K*self.H)/self.delta)
        
        self.SA = pd.MultiIndex.from_product([self.states, self.actions], names = ["s", "a"])
        self.SAS = pd.MultiIndex.from_product([self.states, self.actions, self.states], names = ["s", "a", "s'"])
                    
        self.SA_count = pd.DataFrame(0, index = self.SA, columns = ["count"])
        self.SAS_count = pd.DataFrame(0, index = self.SA, columns = self.states)
        self.SA_reward = pd.DataFrame(0, index = self.SA, columns = ["total"])
        
        self.data = pd.DataFrame(index = range(K), columns = range(self.H), dtype = object)
        
        self.Q = [pd.DataFrame(self.H, index = self.states, columns = self.actions, dtype = float) for _ in range(self.H)]

        self.V = pd.DataFrame(0, index = self.states, columns = range(self.H+1), dtype = float)

        
        self.cumreward = pd.Series(index = range(K), dtype = float)


    def UCB_Q(self):
        P_SAS = self.SAS_count.divide(self.SA_count["count"], axis = 0)
        R = self.SA_reward["total"].divide(self.SA_count["count"])
        for h in range(self.H-1,-1,-1):
            
            Vs = self.V.loc[:,h+1]
            Q = (R + P_SAS.multiply(Vs).sum(axis = 1) + self.bonus(self.SA_count)["count"]).unstack(level = 1).fillna(self.H)
            mask = Q < self.Q[h]
            self.Q[h][mask] = Q[mask]
            self.Q[h].clip(lower = None, upper = self.H, inplace = True)
            
            """
            for sa in self.SA:
                if self.SA_count["count"].loc[sa]:
                    self.Q[h].loc[sa[0], sa[1]] = self.update_q(sa[0], sa[1], h)
                else:
                    self.Q[h].loc[sa[0], sa[1]] = float(self.H)
            
            """
            self.V.loc[:,h] = self.Q[h].max(axis=1)
            
                
                    
    def Bellman_Q(self, state, action, h):
        N_sa = self.SA_count["count"].loc[state,action]
        P_SAS = self.SAS_count.loc[state,action].divide(N_sa)
        Vs = self.V.loc[:, h+1]
        R_sa = self.SA_reward["total"].loc[state, action] / N_sa
        return R_sa + (P_SAS.multiply(Vs)).sum() + self.bonus(N_sa)
        
    def update_q(self, state, action, h):
        return min(self.Q[h].loc[state, action], self.H, self.Bellman_Q(state, action, h))
    
    def bonus(self, N):
        return 7*self.H*self.L*np.sqrt((1/N))
    
    def policy(self, state, actions, h, kwargs):
        Q = kwargs.get('Q', None)
        if Q:
            Qs = Q[h].loc[str(state),:].sample(frac=1.)
            action = Qs.idxmax()
            return literal_eval(action)
        else:
            return (0, 1)
    
    def update_counts(self, episode, SA_count, SAS_count, SA_reward, k):
        """
        for step in episode:
            self.SA_count.loc[str(step[0]), str(step[1])] += 1
            self.SAS_count[str(step[-1])].loc[str(step[0]), str(step[1])] += 1
            self.SA_reward.loc[str(step[0]), str(step[1])] += step[2]
        """
        self.SA_count = self.SA_count + SA_count
        self.SAS_count = self.SAS_count + SAS_count
        self.SA_reward = self.SA_reward + SA_reward
        self.data.loc[k, :] = pd.Series(episode, dtype = object)
        self.cumreward[k] = SA_reward["total"].sum()
            
    def learn(self):
        for k in range(self.K):
            self.UCB_Q()
            episode, SA_count, SAS_count, SA_reward = self.env.sample_episode(self.policy, Q = self.Q)
            self.update_counts(episode, SA_count, SAS_count, SA_reward, k)
        """
        final_policy = []
        for h in range(self.H):
            final_policy.append(self.Q[h].idxmax(axis=1))
        
        return final_policy
        """
        
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
                
                
    
