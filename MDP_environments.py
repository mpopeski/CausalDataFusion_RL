#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
import os



class TabularMDP:
    
    def __init__(self, state_values, action_values, H, n_reward_states = 3,\
                 n_reward_actions = 2, default_prob = 6, policy = "random",\
                 simpson = False):
        
        state_dim = 2
        #action_dim = 2
        
        # get the tabular states
        self.H = H
        self.states = list(product(*[[i for i in range(state_values)] for _ in range(state_dim)]))
        
        # get the actions and mediator values if needed
        moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        if action_values:
            self.actions = list(range(1, action_values+1))
            self.mediators = moves
        else:
            self.actions = moves
            self.mediators = moves
    
        self.max_state = state_values - 1
        self.min_state = 0
        
        self.reward_states = [self.states[i] for i in np.random.choice(len(self.states), replace=False, size = n_reward_states)]
        # change this from reward actions to reward mediators ??
        self.reward_mediators = {self.reward_states[j]: [self.mediators[i] for i in np.random.choice(len(self.mediators),\
                            replace=False, size = n_reward_actions)] for j in range(len(self.reward_states))}
        
        # also need a transition probability from action to mediators
        # if no mediators it's 1 if action == mediator 0 otherwise
        self.act_to_med = self.get_action_to_med(action_values, default_prob)
        
        # change the other functions so transition goes state -> action -> mediator -> next state/reward
        # and make sure it doesn't affect the case when we don't use mediators
        # then we can test the same environment with frontdoor and backdoor
            
        # I have reward distribution over states, u and mediators
        if simpson:
            self.reward_dist = self.get_reward_distribution_simpson()
        else:
            self.reward_dist = self.get_reward_distribution()
        
        self.start_states = [state for state in self.states if state not in self.reward_states] #if sum(state)<=(self.max_state-2)]
        start_prob = np.random.choice(default_prob, replace = True, size = len(self.start_states))
        self.start_prob = np.exp(start_prob) / np.exp(start_prob).sum() 
        
        self.u_prob = {state : (0.8-0.2)*np.random.rand() + 0.2 for state in self.states}
        
        # default policy (obervational) parameters
        if policy == "random":
            self.prob_u1, self.prob_u0 = self.get_default_observational_policy(default_prob)
        elif policy == "v3_eng":
            self.prob_u1, self.prob_u0 = self.get_df_policy_v3(default_prob)
        else:
            raise ValueError("default policy config not specified: Choose random or v3_eng")
        
        # counters
        SA = pd.MultiIndex.from_product([pd.Series(self.states).apply(str), pd.Series(self.actions).apply(str)], 
                                        names = ["s", "a"])
        self.SA_count = pd.DataFrame(0, index = SA, columns = ["count"])
        self.SAS_count = pd.DataFrame(0, index = SA, columns = pd.Series(self.states).apply(str))
        self.SA_reward = pd.DataFrame(0, index = SA, columns = ["total"])
        
    def get_action_to_med(self, action_values, default_prob):
        act_to_med = {}
        if action_values:
            for action in self.actions:
                #prob = np.random.choice(default_prob, size = len(self.mediators), replace = False)
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
    
    def get_reward(self, state, m, conf):
        total_reward = 1
        if not self.reward_dist[state]:
            return 0
        if not self.reward_dist[state][conf][m].any():
            return 0
        
        reward_par = self.reward_dist[state][conf][m]
        rewards = [total_reward - reward_par[0], reward_par[0]]
        idx = np.random.binomial(1, reward_par[1], 1)[0]
        return rewards[idx]

    def start(self):
        id_ = np.random.multinomial(1,self.start_prob,1).argmax()
        initial_state = self.start_states[id_]
        return initial_state
    
    def transition(self, state = None, action = None):
        if state and action: 
            u = np.random.binomial(1, self.u_prob[state], 1)[0]
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
            #will also need to return the mediator later
            return m, reward, next_state
        
        else:
            return self.start()
        
    def transition_conf(self, state = None, action = None, u = None):
        if state and action:
            m_id = np.random.multinomial(1, self.act_to_med[action], 1).argmax()
            #print(m_id)
            m = self.mediators[m_id]
            reward = self.get_reward(state, m, u[0])    
            state_ = np.array(state)
            if np.random.rand() < 0.75:
                if np.random.rand() < 0.5:
                    state_[0] += 2*u-1
                else:
                    state_[1] += 2*u-1
            #print(m)
            next_state = np.clip(state_ + np.array(m), a_min = self.min_state, a_max = self.max_state)
            
            next_state = tuple(next_state)
            #will also need to return the mediator later
            return m, reward, next_state
        
        else:
            return self.start()
    
        
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
    
    def get_reward_distribution_actions(self, act_to_med, reward_dist_med):
        # I would need this just to get observational policy
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
                
                
    
    def get_default_observational_policy(self, default_prob):
        
        total_reward = 1
        initial_logits_u1 = pd.DataFrame(np.array([np.random.choice(default_prob, replace = True, size = len(self.actions)).astype(float)\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
        initial_logits_u0 = pd.DataFrame(np.array([np.random.choice(default_prob, replace = True, size = len(self.actions)).astype(float)\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
        reward_dist = self.get_reward_distribution_actions(self.act_to_med, self.reward_dist)
        for state in self.reward_states:
            
            assert len(initial_logits_u1.loc[str(state)]) == len(self.actions)
            
            max_logit_u1 = initial_logits_u1.loc[str(state)].max()
            max_logit_u0 = initial_logits_u0.loc[str(state)].max()
            
            # best worst case rewards in the reward states
            u0_cont = {key: min(value[0], total_reward - value[0]) for key, value in reward_dist[state][0].items()}
            u1_cont = {key: max(value[0], total_reward - value[0]) for key, value in reward_dist[state][1].items()}
            
            BWC_u0 = max(u0_cont, key = u0_cont.get)
            BWC_u1 = max(u1_cont, key = u1_cont.get)

            idx_action_u0 = self.actions.index(BWC_u0)
            idx_action_u1 = self.actions.index(BWC_u1)
            
            initial_logits_u1.loc[str(state), idx_action_u1] = max_logit_u1 + np.log(2)
            initial_logits_u0.loc[str(state), idx_action_u0] = max_logit_u0 + np.log(2)

        
        prob_u1 = (np.exp(initial_logits_u1).T / np.exp(initial_logits_u1).sum(axis = 1)).T
        prob_u0 = (np.exp(initial_logits_u0).T / np.exp(initial_logits_u0).sum(axis = 1)).T
        
        return pd.DataFrame(prob_u1, index = pd.Series(self.states).apply(str)), pd.DataFrame(prob_u0, index = pd.Series(self.states).apply(str))
        
    
    def get_df_policy_v3(self, default_prob):
        
        initial_logits_u1 = pd.DataFrame(np.array([np.random.choice(default_prob, replace = True, size = len(self.actions)).astype(float)\
                                                   for _ in range(len(self.states))]), index = pd.Series(self.states).apply(str))
        initial_logits_u0 = pd.DataFrame(np.array([np.random.choice(default_prob, replace = True, size = len(self.actions)).astype(float)\
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
            
            initial_logits_u1.loc[str(state), idx_action_u1] = max_logit_u1 + np.log(4)
            initial_logits_u0.loc[str(state), idx_action_u0] = max_logit_u0 + np.log(4)

        
        prob_u1 = (np.exp(initial_logits_u1).T / np.exp(initial_logits_u1).sum(axis = 1)).T
        prob_u0 = (np.exp(initial_logits_u0).T / np.exp(initial_logits_u0).sum(axis = 1)).T
        
        return pd.DataFrame(prob_u1, index = pd.Series(self.states).apply(str)), pd.DataFrame(prob_u0, index = pd.Series(self.states).apply(str))
    
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
        
        
        def default_val1():
            return pd.DataFrame(0, index = self.SA_count.index, columns = ["count"])
        def default_val2():
            return pd.DataFrame(0, index = self.SAS_count.index, columns = self.SAS_count.columns)
        # NEW COUNTS needed for frontdoor criterion
        # SAM_count
        SAM_count  = defaultdict(default_val1)
        # SAMS_count
        SAMS_count = defaultdict(default_val2)
        # SAM_reward - sum of rewards with state s,  action a, mediator m
        SAM_reward = defaultdict(default_val1)
        
        for k in range(K):
            episode = []
            s = self.start()
            for h in range(self.H):
                u = np.random.binomial(1, self.u_prob[s], 1)
                a = self.default_policy(s, self.actions, h, u)
                m, r, s_ = self.transition_conf(s, a, u)
                
                S_count.loc[str(s)] += 1
                SA_count.loc[str(s),str(a)] += 1
                SAS_count[str(s_)].loc[str(s),str(a)] += 1
                SA_reward.loc[str(s),str(a)] += r
                
                SU_count.loc[str(s), u] += 1
                SUA_count[u[0]].loc[str(s), str(a)] += 1
                SUA_reward[u[0]].loc[str(s), str(a)] += r
                SUAS_count[u[0]][str(s_)].loc[str(s), str(a)] += 1
                
                SAM_count[m].loc[str(s), str(a)] += 1
                SAMS_count[m][str(s_)].loc[str(s), str(a)] += 1
                SAM_reward[m].loc[str(s), str(a)] += r
                
                episode.append((s,u,a,r,s_))
                
                s = s_
            episodes.append(episode)
        
        counts = {"SA": SA_count, "SAS": SAS_count, "SA_reward": SA_reward, "S": S_count, 
                  "SU": SU_count, "SUA": SUA_count,"SUA_reward": SUA_reward, "SUAS": SUAS_count,
                  "SAM": SAM_count, "SAMS": SAMS_count, "SAM_reward": SAM_reward}
        
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
        
