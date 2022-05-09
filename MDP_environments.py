#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product

"""
3 environments:
    1. Causal Gridworld
    
    2. fully factored environment groups of state variables at time t+1 are only affected by
    the same group at time t, and actions only affect specific groups. i.e. no crossover
    
    3. sparse environment where state variables at time t+1 depend on random 
    amount of state variables at time t, (same for actions), there can be crossover
    i.e. for example one variable in time t can have influence on 2 or more variable
    groups in time t+1
"""

class TabularMDP:
    
    def __init__(self, state_dim, state_values, action_dim, action_values, H, n_rewards = 2, default_prob = 10):
        
        if isinstance(action_values, int):
            action_values = [i for i in range(action_values)]
        
        assert isinstance(action_values, list)
        assert action_dim == state_dim

        self.H = H
        self.states = list(product(*[[i for i in range(state_values)] for _ in range(state_dim)]))
        actions = list(product(*[[i for i in action_values] for _ in range(action_dim)]))
        self.actions = [action for action in actions if (np.array(action) == 0).sum() == (action_dim - 1)]
    
        self.max_state = state_values - 1
        self.min_state = 0
        
        
        
        self.reward_states = [self.states[i] for i in np.random.choice(len(self.states), replace=False, size = n_rewards)]
        
        self.start_states = [state for state in self.states if state not in self.reward_states] #if sum(state)<=(self.max_state-2)]
        start_prob = np.random.choice(default_prob, replace = True, size = len(self.start_states))
        self.start_prob = np.exp(start_prob) / np.exp(start_prob).sum() 
        
        self.u_prob = {state : (0.8-0.2)*np.random.rand() + 0.2 for state in self.states}
        
        # default policy (obervational) parameters
        default_prob = np.array([np.random.choice(default_prob, replace = True, size = len(self.actions)) for _ in range(len(self.states))])
        self.p1 = pd.DataFrame((np.exp(default_prob).T / np.exp(default_prob).sum(axis = 1)).T,
                               index = pd.Series(self.states).apply(str))
            
        self.p2 = pd.DataFrame((np.exp(-default_prob).T / np.exp(-default_prob).sum(axis = 1)).T,
                               index = pd.Series(self.states).apply(str))
        # counters
        SA = pd.MultiIndex.from_product([pd.Series(self.states).apply(str), pd.Series(self.actions).apply(str)], 
                                        names = ["s", "a"])
        self.SA_count = pd.DataFrame(0, index = SA, columns = ["count"])
        self.SAS_count = pd.DataFrame(0, index = SA, columns = pd.Series(self.states).apply(str))
        self.SA_reward = pd.DataFrame(0, index = SA, columns = ["total"])
        
    def transition(self, state = None, action = None):
        
        if state and action: 
            reward = 0
            if state in self.reward_states:
                reward = 0.1
                
            u = np.random.binomial(1, self.u_prob[state], 1)
            state_ = np.array(state)
            if np.random.rand() < 0.75:
                if np.random.rand() < 0.5:
                    state_[0] += 2*u-1
                else:
                    state_[1] += 2*u-1
            next_state = np.clip(state_ + np.array(action), a_min = self.min_state, a_max = self.max_state)
            next_state = tuple(next_state)
            return reward, next_state
        
        else:
            return self.start()
        
    def transition_conf(self, state = None, action = None, u = None):
        
        if state and action: 
            reward = 0
            if state in self.reward_states:
                reward = 0.1
                
            state_ = np.array(state)
            if np.random.rand() < 0.75:
                if np.random.rand() < 0.5:
                    state_[0] += 2*u-1
                else:
                    state_[1] += 2*u-1
            next_state = np.clip(state_ + np.array(action), a_min = self.min_state, a_max = self.max_state)
            next_state = tuple(next_state)
            return reward, next_state
        
        else:
            return self.start()
    
    def start(self):
        id_ = np.random.multinomial(1,self.start_prob,1).argmax()
        initial_state = self.start_states[id_]
        return initial_state
    
    def sample_episode(self, policy, **kwargs):
        episode = []
        s = self.start()
        self.SA_count[:] = 0
        self.SAS_count[:] = 0
        self.SA_reward[:] = 0
        for h in range(self.H):
            a = policy(s, self.actions, h, kwargs)
            r, s_ = self.transition(s,a)
            self.SA_count.loc[str(s),str(a)] += 1
            self.SAS_count[str(s_)].loc[str(s),str(a)] += 1
            self.SA_reward.loc[str(s),str(a)] += r
            episode.append((s,a,r,s_))
            s = s_
        return episode, self.SA_count, self.SAS_count, self.SA_reward
    
    def default_policy(self, s, actions, h, u=0):
        if u:
            prob = self.p1.loc[str(s), :]
        else:
            prob = self.p2.loc[str(s), :]
        idx = np.random.choice(len(actions), size = 1, p = prob)
        return actions[idx[0]]
    
    def get_obs_data(self, K):
        episodes = []
        SA_count = pd.DataFrame(0, index = self.SA_count.index, columns = self.SA_count.columns)
        SAS_count = pd.DataFrame(0, index = self.SAS_count.index, columns = self.SAS_count.columns)
        SA_reward = pd.DataFrame(0, index = self.SA_reward.index, columns = self.SA_reward.columns)
        S_count = pd.DataFrame(0, index = pd.Series(self.states).apply(str), columns = ["count"])
        SU_count = pd.DataFrame(0, index = pd.Series(self.states).apply(str), columns = [0, 1])
        SUA_count = [pd.DataFrame(0, index = self.SA_count.index, columns = ["count"]),
                     pd.DataFrame(0, index = self.SA_count.index, columns = ["count"])]
        SUAS_count = [pd.DataFrame(0, index = self.SAS_count.index, columns = self.SAS_count.columns),
                      pd.DataFrame(0, index = self.SAS_count.index, columns = self.SAS_count.columns)]
        
        for k in range(K):
            episode = []
            s = self.start()
            for h in range(self.H):
                u = np.random.binomial(1, self.u_prob[s], 1)
                a = self.default_policy(s, self.actions, h, u)
                r, s_ = self.transition_conf(s, a, u)
                
                S_count.loc[str(s)] += 1
                SA_count.loc[str(s),str(a)] += 1
                SAS_count[str(s_)].loc[str(s),str(a)] += 1
                SA_reward.loc[str(s),str(a)] += r
                SU_count.loc[str(s), u] += 1
                SUA_count[u[0]].loc[str(s), str(a)] += 1
                SUAS_count[u[0]][str(s_)].loc[str(s), str(a)] += 1
                episode.append((s,u,a,r,s_))
                
                s = s_
            episodes.append(episode)
        
        counts = {"SA": SA_count, "SAS": SAS_count, "SA_reward": SA_reward,
                  "S": S_count, "SU": SU_count, "SUA": SUA_count, "SUAS": SUAS_count}
        
        return pd.DataFrame(episodes, dtype = object), counts
    
    def observational_data(self, K):
        self.data, self.counts = self.get_obs_data(K) 
                

class causalBlock:
    
    def __init__(self, Sn, Sm, An, Am):
        pass
    
    def transition(self, state, action):
        pass

class sparseMDP:
    
    def __init__(self):
        pass
    
    def get_states(self, args):
        """
        get the state space based on arguments such as:
            1. the number of variables Sn
            2. the number of values each variable can take Sm
        """
        pass
    
    def get_actions(self, args):
        """
        get the action space based argumnes:
            1. the number of action variables (subactions) An
            2. the number of values each action can take Am
        """
        pass
    
    def get_causal_structure(self, args):
        """
        get the rules of how the transition probability will factorize
        should return some form of dictonary or dataframe that can later can also be used
        in the learning algorithms
        """
        pass
    
    def transition(self, args):
        """
        define a transition probability based on the causal structure
        """
        pass
    
    def reward(self, state, action):
        """
        define a reward function, maybe in this case as a sum of the individual factorizations
        """
        pass
    
    def play(self, state, action):
        """
        takes input current state and action
        returns next state, reward (or rewards depending on assumption)
        """
        pass
    