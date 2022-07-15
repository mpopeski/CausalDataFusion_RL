#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
import os


class TabularMDP:
    
    def __init__(self, state_values = 3, action_values = 0, H = 5, n_reward_states = 3,\
                 n_reward_actions = 2, default_prob = 6, policy = "random",\
                 simpson = False, conf_values = 2):
        
        state_dim = 2
        #action_dim = 2
        
        self.H = H
        # get the tabular states
        self.states = list(product(*[[i for i in range(state_values)] for _ in range(state_dim)]))
        
        # get the actions and mediator values(if needed)
        moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        if action_values:
            self.actions = list(range(1, action_values+1))
            self.mediators = moves
        else:
            self.actions = moves
            self.mediators = moves
    
        self.max_state = state_values - 1
        self.min_state = 0
        
        # get random reward states 
        self.reward_states = [self.states[i] for i in np.random.choice(len(self.states), replace=False, size = n_reward_states)]
        # get reward mediators (it is the same as reward action when no mediators are used)
        self.reward_mediators = {reward_state: [self.mediators[i] for i in np.random.choice(len(self.mediators),\
                            replace=False, size = n_reward_actions)] for reward_state in self.reward_states}
        
        # transition probability from action to mediators
        # if no mediators the probability is 1 if action == mediator 0 otherwise
        self.act_to_med = self.get_action_to_med(action_values, default_prob)
        
        # I have reward distribution over states, u and mediators
        if simpson:
            self.reward_dist = self.get_reward_distribution_simpson()
        else:
            self.reward_dist = self.get_reward_distribution()
        
        # get probabilities for starting position, it is 0 for reward states
        self.start_states = [state for state in self.states if state not in self.reward_states]
        start_prob = default_prob * np.random.rand(len(self.start_states)) / 2
        self.start_prob = np.exp(start_prob) / np.exp(start_prob).sum() 
        

        assert conf_values >= 2
        
        self.observable_conf = list(range(conf_values))
        self.n_conf = conf_values
        if conf_values == 2:
            # binary confounder
            self.u_prob = {state : (0.8-0.2)*np.random.rand() + 0.2 for state in self.states}
        else:
            self.get_confounder_dist(conf_values)
        
        # behavioral policy (obervational) parameters
        if policy == "random":
            self.prob_u1, self.prob_u0 = self.get_default_observational_policy(default_prob)
        elif policy == "v3_eng":
            self.prob_u1, self.prob_u0 = self.get_df_policy_v3(default_prob)
        else:
            raise ValueError("default policy config not specified: Choose random or v3_eng")
        

        self.S_index = pd.Series(self.states).apply(str)
        self.SA_index = pd.MultiIndex.from_product([self.S_index, pd.Series(self.actions).apply(str)], 
                                                   names = ["s", "a"])
        self.SU_index = pd.MultiIndex.from_product([self.S_index, pd.Series(self.observable_conf).apply(str)],
                                                   names = ["s", "u"])
        self.SUA_index = pd.MultiIndex.from_product([self.S_index, pd.Series(self.observable_conf).apply(str), pd.Series(self.actions).apply(str)],
                                                    names = ["s", "u", "a"])
        self.SAM_index = pd.MultiIndex.from_product([self.S_index, pd.Series(self.actions).apply(str), pd.Series(self.mediators).apply(str)],
                                                    names = ["s", "a", "m"])

    def start(self):
        # get a starting state
        id_ = np.random.multinomial(1,self.start_prob,1).argmax()
        initial_state = self.start_states[id_]
        return initial_state
    
    def get_confounder(self, state):
        # get the confounder values both observable and unbservable used in transition and reward function
        u = np.random.multinomial(1, self.u_prob[state],1).argmax()
        par = self.w_par[state][u]
        w1 = np.random.binomial(1,(2*par[0] + par[1])/3,1)[0]
        w2 = np.random.binomial(1,(2*par[1] + par[0])/3,1)[0]
        return u, w1, w2
    
    def get_reward(self, state, m, conf):
        # get reward from the reward distribution
        total_reward = 1
        if not self.reward_dist[state]:
            return 0
        if not self.reward_dist[state][conf][m].any():
            return 0
        
        reward_par = self.reward_dist[state][conf][m]
        rewards = [total_reward - reward_par[0], reward_par[0]]
        idx = np.random.binomial(1, reward_par[1], 1)[0]
        return rewards[idx]
    
    def transition(self, state = None, action = None):
        # online transition function
        if state and action:
            if self.n_conf == 2:
                u = np.random.binomial(1, self.u_prob[state], 1)[0]
                w1 = u
                w2 = u
            else:
                u, w1, w2 = self.get_confounder(state)
                
            m_id = np.random.multinomial(1, self.act_to_med[action], 1).argmax()
            m = self.mediators[m_id]
            
            reward = self.get_reward(state, m, w2)
            
            state_ = np.array(state)
            if np.random.rand() < 0.75:
                if np.random.rand() < 0.5:
                    state_[0] += 2*w2-1
                else:
                    state_[1] += 2*w2-1
                    
            next_state = np.clip(state_ + np.array(m), a_min = self.min_state, a_max = self.max_state)
            next_state = tuple(next_state)
            return m, reward, next_state
        
        else:
            return self.start()
        
    def transition_conf(self, state = None, action = None, u = None):
        # offline transition function
        if state and action:
            m_id = np.random.multinomial(1, self.act_to_med[action], 1).argmax()
            m = self.mediators[m_id]
            
            reward = self.get_reward(state, m, u)
            
            state_ = np.array(state)
            if np.random.rand() < 0.75:
                if np.random.rand() < 0.5:
                    state_[0] += 2*u-1
                else:
                    state_[1] += 2*u-1

            next_state = np.clip(state_ + np.array(m), a_min = self.min_state, a_max = self.max_state)
            next_state = tuple(next_state)
            return m, reward, next_state
        
        else:
            return self.start()
        
    def default_policy(self, state, actions, u=0):
        # get an action from the behavioral policy
        if u:
            prob = self.prob_u1.loc[str(state), :]            
        else:
            prob = self.prob_u0.loc[str(state), :]
        idx = np.random.choice(len(actions), size = 1, p = prob)
        return actions[idx[0]]
    
    def get_obs_data(self, K):
        # get the observational data collected by episodes
        # the columns of the dataframe are current state, action, reward, next_state
        # plus confounder or intermediate variable depending on the scenario
        
        data = []
        for k in range(K):
            s = self.start()
            for h in range(self.H):
                # the observational agent observes the confounder
                
                if self.n_conf == 2:
                    u = np.random.binomial(1, self.u_prob[s], 1)[0]
                    w1 = u
                    w2 = u
                else:
                    u, w1, w2 = self.get_confounder(s)
                    
                a = self.default_policy(s, self.actions, w1)
                m, r, s_ = self.transition_conf(s, a, w2)
                
                data.append((s,u,a,m,r,s_))
                s = s_
        
        data = pd.DataFrame(data, columns= ["s","u","a","m","r","s_"]).applymap(str)
        
        data.loc[:,"r"] = data.loc[:,"r"].apply(float)
        return data
   
    def observational_data(self, K):
        self.data = self.get_obs_data(K)
    
    #HELFPER FUNCs: used to define all parts of the environment
    
    def get_confounder_dist(self, obs_k):
        # Used to define structure of the confounding
        self.u_prob = {state:self.get_multinomial_prob(obs_k) for state in self.states}
        self.w_par = {state:self.bin_probs(obs_k) for state in self.states}
        
    def get_action_to_med(self, action_values, default_prob):
        # Takes input discrete action space and maps the discrete actions to
        # a mediator which is a move - {left, right, up, down}
        # if action_values = 0, then actions = moves and we have no mediators
        
        act_to_med = {}
        if action_values:
            for action in self.actions:
                prob = default_prob * np.random.rand(len(self.mediators))
                prob = np.exp(prob) / np.exp(prob).sum()
                act_to_med[action] = prob
        else:
            for action in self.actions:
                prob = np.zeros(len(self.mediators))
                for i, mediator in enumerate(self.mediators):
                    if action == mediator:
                        prob[i] = 1
                act_to_med[action] = prob
        
        return act_to_med
    
    def get_reward_distribution(self):
        # Option #1: basic reward distribution for a binary confounder
        
        total_reward = 1
        reward_dist = defaultdict(dict)
        
        def default_val():
            return np.array([])
        
        for state,  mediators in self.reward_mediators.items():
            dist = {}
            dist[0] = defaultdict(default_val)
            dist[1] = defaultdict(default_val)
            for mediator in mediators:
                conf_rewards = (total_reward / 2) * np.random.rand(2)
                conf_prob = (conf_rewards.max() + 0.1)*np.random.rand(2)
                dist[0][mediator] = np.array([conf_rewards[0], conf_prob[0]])
                dist[1][mediator] = np.array([conf_rewards[1], conf_prob[1]])
            
            reward_dist[state] = dist
        return reward_dist
    
    def get_reward_distribution_simpson(self):
        # Option #2: reward distribution to induce the simpson paradox in RL
        
        reward_dist = defaultdict(dict)
        
        def default_val():
            return np.array([])
        
        for state,  mediators in self.reward_mediators.items():
            dist = {}
            dist[0] = defaultdict(default_val)
            dist[1] = defaultdict(default_val)
            rewards_lower = 0.5*np.random.rand(len(mediators))
            rewards_lower.sort()
            rewards_upper = 0.1*np.random.rand(len(mediators)) + 0.8
            rewards_upper.sort()
            mediators_ = pd.Series(mediators).sample(frac=1.)
            
            for i in range(len(mediators_)):
                dist[0][mediators_.iloc[i]] = np.array([rewards_lower[i], 1])
                dist[1][mediators_.iloc[i]] = np.array([rewards_upper[i], 1])
                
            reward_dist[state] = dist
        return reward_dist
               
    def get_default_observational_policy(self, default_prob):
        # Behavioureal policy #1: basic random observational policy

        initial_logits_u1 = pd.DataFrame(np.array([1 * np.random.rand(len(self.actions))\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
        initial_logits_u0 = pd.DataFrame(np.array([1 * np.random.rand(len(self.actions))\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
 
        prob_u1 = (np.exp(initial_logits_u1).T / np.exp(initial_logits_u1).sum(axis = 1)).T
        prob_u0 = (np.exp(initial_logits_u0).T / np.exp(initial_logits_u0).sum(axis = 1)).T
        return pd.DataFrame(prob_u1, index = pd.Series(self.states).apply(str)), pd.DataFrame(prob_u0, index = pd.Series(self.states).apply(str))
        
    
    def get_df_policy_v3(self, default_prob):
        # Behavioural policy #2: Engineered policy to induce the simpson paradox in RL
        
        initial_logits_u1 = pd.DataFrame(np.array([(default_prob-1) * np.random.rand(len(self.actions)) + 1\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
        initial_logits_u0 = pd.DataFrame(np.array([(default_prob-1) * np.random.rand(len(self.actions)) + 1\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
        
        reward_dist = self.get_reward_distribution_actions(self.act_to_med, self.reward_dist)
        for state in self.reward_states:
            
            assert len(initial_logits_u1.loc[str(state)]) == len(self.actions)
            
            max_logit_u1 = initial_logits_u1.loc[str(state)].max()
            max_logit_u0 = initial_logits_u0.loc[str(state)].max()
            
            # best case rewards in the lower reward confounder
            u0_cont = {key: value[0] for key, value in reward_dist[state][0].items()}
            # worst case rewards in the uppder confounder
            u1_cont = {key: value[0] for key, value in reward_dist[state][1].items()}
            
            BWC_u0 = max(u0_cont, key = u0_cont.get)
            BWC_u1 = min(u1_cont, key = u1_cont.get)

            idx_action_u0 = self.actions.index(BWC_u0)
            idx_action_u1 = self.actions.index(BWC_u1)
            
            initial_logits_u1.loc[str(state), idx_action_u1] = max_logit_u1 + np.log(2)
            initial_logits_u0.loc[str(state), idx_action_u0] = max_logit_u0 + np.log(2)

        
        prob_u1 = (np.exp(initial_logits_u1).T / np.exp(initial_logits_u1).sum(axis = 1)).T
        prob_u0 = (np.exp(initial_logits_u0).T / np.exp(initial_logits_u0).sum(axis = 1)).T
        return pd.DataFrame(prob_u1, index = pd.Series(self.states).apply(str)), pd.DataFrame(prob_u0, index = pd.Series(self.states).apply(str))
    
    # UTILITY FUNCs: used in the helper functions
    
    def get_reward_distribution_actions(self, act_to_med, reward_dist_med):
        # Utility function used in behavioural policy #2
        
        reward_dist = defaultdict(dict)
        
        def default_val():
            return np.array([])
        
        for state in self.reward_states:
            dist = {}
            dist[0] = defaultdict(default_val)
            dist[1] = defaultdict(default_val)
            rewards0 = reward_dist_med[state][0]
            rewards1 = reward_dist_med[state][1]
            for action in self.actions:
                r0 = 0
                r1 = 0
                for med in rewards0.keys():
                    r0 += (rewards0[med][0]*rewards0[med][1] + (1-rewards0[med][0])*(1-rewards0[med][1]))*\
                        act_to_med[action][self.mediators.index(med)]
                    r1 += (rewards1[med][0]*rewards1[med][1] + (1-rewards1[med][0])*(1-rewards1[med][1]))*\
                        act_to_med[action][self.mediators.index(med)]
                if r0 > 0 or r1 > 0:
                    dist[0][action] = np.array([r0, 1])
                    dist[1][action] = np.array([r1, 1])
            
            reward_dist[state] = dist
        return reward_dist
            
    def get_multinomial_prob(self, obs_k):
        # Utility function to get probabilities for multinomial distribution
        # used in to define confounder distribution
        
        base = 1 / obs_k
        add = 2*np.random.randint(low=0, high = 2, size = obs_k)-1
        val = (1/(2*obs_k))*np.random.rand(obs_k)
        prob_ = base + add*val
        prob = prob_/prob_.sum()
        
        return prob
    
    def bin_probs(self, obs_k):
        # get a parameter for binomial distributions for different parent values
        # used in to define confounder distribution
        par = []
        for _ in range(obs_k):
            par.append(((0.8-0.2)*np.random.rand() + 0.2, (0.8-0.2)*np.random.rand() + 0.2))
        return par
    
    
    #TODO: Update the save_env function and create new one load_env to guarantee reproducibility of results
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
        
