#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
import os



class TabularMDP:
    
    def __init__(self, state_dim, state_values, action_dim, action_values, H, n_reward_states = 3, n_reward_actions = 2, default_prob = 6, policy = "random"):
        
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
        
        self.reward_states = [self.states[i] for i in np.random.choice(len(self.states), replace=False, size = n_reward_states)]
        self.reward_actions = {self.reward_states[j]: [self.actions[i] for i in np.random.choice(len(self.actions),\
                            replace=False, size = n_reward_actions)] for j in range(len(self.reward_states))}
        self.reward_dist = self.get_reward_distribution()
        
        self.start_states = [state for state in self.states if state not in self.reward_states] #if sum(state)<=(self.max_state-2)]
        start_prob = np.random.choice(default_prob, replace = True, size = len(self.start_states))
        self.start_prob = np.exp(start_prob) / np.exp(start_prob).sum() 
        
        self.u_prob = {state : (0.8-0.2)*np.random.rand() + 0.2 for state in self.states}
        
        # default policy (obervational) parameters
        if policy == "random":
            self.prob_u1, self.prob_u0 = self.get_default_observational_policy(default_prob)
        elif policy == "v2_eng":
            self.prob_u1, self.prob_u0 = self.get_df_policy_v2(default_prob)
        
        # counters
        SA = pd.MultiIndex.from_product([pd.Series(self.states).apply(str), pd.Series(self.actions).apply(str)], 
                                        names = ["s", "a"])
        self.SA_count = pd.DataFrame(0, index = SA, columns = ["count"])
        self.SAS_count = pd.DataFrame(0, index = SA, columns = pd.Series(self.states).apply(str))
        self.SA_reward = pd.DataFrame(0, index = SA, columns = ["total"])
        
    def transition(self, state = None, action = None):
        if state and action: 
            u = np.random.binomial(1, self.u_prob[state], 1)[0]
            reward = self.get_reward(state, action, u)
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
            reward = self.get_reward(state, action, u[0])    
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
    
    def get_reward_distribution(self):
        
        total_reward = 1
        reward_dist = defaultdict(dict)
        
        def default_val():
            return np.array([])
        
        for state,  actions in self.reward_actions.items():
            dist = {}
            dist[0] = defaultdict(default_val)
            dist[1] = defaultdict(default_val)
            for action in actions:
                conf_rewards = (total_reward / 2) * np.random.rand(2)
                conf_prob = (conf_rewards.max() + 0.1)*np.random.rand(2)
                dist[0][action] = np.array([conf_rewards[0], conf_prob[0]])
                dist[1][action] = np.array([conf_rewards[1], conf_prob[1]])
            
            reward_dist[state] = dist
        
        return reward_dist
    
    def get_default_observational_policy(self, default_prob):
        
        total_reward = 1
        initial_logits_u1 = pd.DataFrame(np.array([np.random.choice(default_prob, replace = True, size = len(self.actions)).astype(float)\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
        initial_logits_u0 = pd.DataFrame(np.array([np.random.choice(default_prob, replace = True, size = len(self.actions)).astype(float)\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
            
        for state in self.reward_states:
            
            assert len(initial_logits_u1.loc[str(state)]) == len(self.actions)
            
            max_logit_u1 = initial_logits_u1.loc[str(state)].max()
            max_logit_u0 = initial_logits_u0.loc[str(state)].max()
            
            # best worst case rewards in the reward states
            u0_cont = {key: min(value[0], total_reward - value[0]) for key, value in self.reward_dist[state][0].items()}
            u1_cont = {key: min(value[0], total_reward - value[0]) for key, value in self.reward_dist[state][1].items()}
            
            BWC_u0 = max(u0_cont, key = u0_cont.get)
            BWC_u1 = max(u1_cont, key = u1_cont.get)

            idx_action_u0 = self.actions.index(BWC_u0)
            idx_action_u1 = self.actions.index(BWC_u1)
            
            initial_logits_u1.loc[str(state), idx_action_u1] = max_logit_u1 + np.log(2)
            initial_logits_u0.loc[str(state), idx_action_u0] = max_logit_u0 + np.log(2)

        
        prob_u1 = (np.exp(initial_logits_u1).T / np.exp(initial_logits_u1).sum(axis = 1)).T
        prob_u0 = (np.exp(initial_logits_u0).T / np.exp(initial_logits_u0).sum(axis = 1)).T
        
        return pd.DataFrame(prob_u1, index = pd.Series(self.states).apply(str)), pd.DataFrame(prob_u0, index = pd.Series(self.states).apply(str))
        
    def get_df_policy_v2(self, default_prob):
        total_reward = 1
        prob_u1 = pd.DataFrame(0, index = pd.Series(self.states).apply(str), columns = range(4))
        prob_u0 = pd.DataFrame(0, index = pd.Series(self.states).apply(str), columns = range(4))
        n_reward_actions = len(list(self.reward_actions.values())[0])
        total_actions = len(self.actions)
        assert len(prob_u1.iloc[0]) == total_actions
        assert len(prob_u0.iloc[0]) == total_actions
        
        for state in self.reward_states:
            
            # best worst case rewards in the reward states
            u0_cont = {key: min(value[0], total_reward - value[0]) for key, value in self.reward_dist[state][0].items()}
            u1_cont = {key: min(value[0], total_reward - value[0]) for key, value in self.reward_dist[state][1].items()}
            
            
            BWC_u0 = max(u0_cont, key = u0_cont.get)
            BWC_u1 = max(u1_cont, key = u1_cont.get)
            
            reward_actions = list(u0_cont.keys())
            #reward_actions1 = list(u1_cont.keys())

            
            prob0 = (0.1/(total_actions - n_reward_actions)) * np.ones(total_actions)
            prob1 = (0.1/(total_actions - n_reward_actions)) * np.ones(total_actions)
            for action in reward_actions: #0, reward_actions1):
                idx_action = self.actions.index(action)
                if action == BWC_u0:
                    prob0[idx_action] = 0.6
                else:
                    prob0[idx_action] = 0.3/ (n_reward_actions - 1)
                    
                if action == BWC_u1:
                    prob1[idx_action] = 0.75
                else:
                    prob1[idx_action] = 0.15/ (n_reward_actions - 1)

                    
            prob_u0.loc[str(state), :] = prob0
            prob_u1.loc[str(state), :] = prob1
        
        for state in self.start_states:
            counter = 1000000
            closest_rw = None
            for rw_state in self.reward_states:
                distance = abs(state[0] - rw_state[0]) + abs(state[1] - rw_state[1]) 
                if distance < counter:
                    counter = distance
                    closest_rw = rw_state
            x_action = (np.sign(closest_rw[0] - state[0]), 0)
            y_action = (0, np.sign(closest_rw[1] - state[1]))
            
            logits = np.random.choice(default_prob, replace = True, size = len(self.actions)).astype(float)
            logits0 = logits.copy()
            logits1 = logits.copy()
            for i in range(total_actions):
                if self.actions[i] == x_action or self.actions[i] == y_action:
                    logits0[i] = logits.max() + np.log(2) 
                    logits1[i] = logits.max() + np.log(4)
            prob_u0.loc[str(state), :]= np.exp(logits0) / np.exp(logits0).sum()
            prob_u1.loc[str(state), :]= np.exp(logits1) / np.exp(logits1).sum()        
            
        return prob_u1, prob_u0
    
    def get_reward(self, state, action, conf):
        total_reward = 1
        if self.reward_dist[state]:
            if self.reward_dist[state][conf][action].any():
                reward_par = self.reward_dist[state][conf][action]
                rewards = [reward_par[0], total_reward-reward_par[0]]
                idx = np.random.binomial(1, reward_par[1], 1)[0]
                return rewards[idx]
            return 0
        return 0
    
    def default_policy(self, state, actions, h, u=0):
        if u:
            prob = self.prob_u1.loc[str(state), :]            
        else:
            prob = self.prob_u0.loc[str(state), :]
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
        SUA_reward = [pd.DataFrame(0, index = self.SA_count.index, columns = ["total"]),
                     pd.DataFrame(0, index = self.SA_count.index, columns = ["total"])]
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
                SUA_reward[u[0]].loc[str(s), str(a)] += r
                SUAS_count[u[0]][str(s_)].loc[str(s), str(a)] += 1
                episode.append((s,u,a,r,s_))
                
                s = s_
            episodes.append(episode)
        
        counts = {"SA": SA_count, "SAS": SAS_count, "SA_reward": SA_reward, "S": S_count, 
                  "SU": SU_count, "SUA": SUA_count,"SUA_reward":SUA_reward, "SUAS": SUAS_count}
        
        return pd.DataFrame(episodes, dtype = object), counts
    
    def observational_data(self, K):
        self.data, self.counts = self.get_obs_data(K) 
        
    def save_env(self, path):
        os.makedirs(path, exist_ok=True)
        index = []
        reward0 = []
        prob0 = []
        reward1 = []
        prob1 = []
        for state, actions in self.reward_actions.items():
            for action in actions:
                index.append((str(state), str(action)))
                reward0.append(self.reward_dist[state][0][action][0])
                prob0.append(self.reward_dist[state][0][action][1])
                reward1.append(self.reward_dist[state][1][action][0])
                prob1.append(self.reward_dist[state][1][action][1])
        
        index = pd.MultiIndex.from_tuples(index, names = ["s","a"])
        reward_dist = pd.DataFrame(np.array([reward0, prob0, reward1, prob1]).T, index = index, columns = ["reward0", "prob0", "reward1", "prob1"])
        reward_dist.to_csv(path + "reward_dist.csv")
        pd.DataFrame(self.u_prob.values(), index = pd.Series(self.u_prob.keys()).apply(str), columns = ["prob of 1"]).to_csv(path + "confounder.csv")
        prob_u0 = self.prob_u0.copy()
        prob_u1 = self.prob_u1.copy()
        prob_u0.columns = pd.Series(self.actions).apply(str)
        prob_u1.columns = pd.Series(self.actions).apply(str)
        policy_folder = path + "policy/"
        os.makedirs(policy_folder, exist_ok=True)
        prob_u0.to_csv(path + "policy/actions_u0.csv")
        prob_u1.to_csv(path + "policy/actions_u1.csv")
        pd.DataFrame(self.start_prob, index = pd.Series(self.start_states).apply(str), columns = ["start_prob"]).to_csv(path +"start_prob.csv")
        
