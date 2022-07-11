#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def naive_strategy(obs_data, S_index, SA_index):
    
    N_sa = pd.Series(0, index = SA_index)
    N_sa.update(obs_data[["s","a","s_"]].groupby(["s","a"]).count()["s_"])
    
    N_sas = pd.DataFrame(0, index = SA_index, columns = S_index)
    N_sas.update(obs_data[["s","a","s_","m"]].groupby(["s","a","s_"]).count()["m"].unstack(level = -1))
    
    SA_reward = pd.Series(0, index = SA_index)
    SA_reward.update(obs_data[["s","a","r"]].groupby(["s","a"]).sum()["r"])
    
    R_sa = SA_reward.divide(N_sa.clip(lower = 1, upper = None), axis = 0)
    P_sas = N_sas.divide(N_sa.clip(lower = 1, upper = None), axis = 0)
    
    return N_sa, R_sa, P_sas


def backdoor_strategy(obs_data, S_index, SU_index, SUA_index):
    
    N_s = pd.Series(0, index = S_index)
    N_s.update(obs_data[["s","a"]].groupby("s").count()["a"])
    
    N_su = pd.Series(0, index = SU_index)
    N_su.update(obs_data[["s","u","a"]].groupby(["s","u"]).count()["a"])
    
    N_sua = pd.Series(0, index = SUA_index)
    N_sua.update(obs_data[["s","u","a","s_"]].groupby(["s","u","a"]).count()["s_"])
    
    N_suas = pd.DataFrame(0, index = SUA_index, columns = S_index)
    N_suas.update(obs_data[["s","u","a","s_","m"]].groupby(["s","u","a","s_"]).count()["m"].unstack(level = -1))
    
    r_sua = pd.Series(0, index = SUA_index)
    r_sua.update(obs_data[["s","u","a","r"]].groupby(["s","u","a"]).sum()["r"])
    
    P_su = N_su.divide(N_s.clip(lower = 1, upper = None), axis = 0, level=0)
    P_suas = N_suas.divide(N_sua.clip(lower = 1, upper = None), axis = 0)
    R_sua = r_sua.divide(N_sua.clip(lower = 1, upper = None), axis = 0)
    
    P_sas = P_suas.multiply(P_su, axis = 0).groupby(level = [0,-1]).sum()
    R_sa = R_sua.multiply(P_su, axis = 0).groupby(level = [0,-1]).sum() 
    N_sa = N_sua.groupby(level=[0,-1]).min()
    
    return N_sa, R_sa, P_sas

def frontdoor_strategy(obs_data, S_index, SA_index, SAM_index):
    
    N_s = pd.Series(0, index = S_index)
    N_s.update(obs_data[["s","a"]].groupby("s").count()["a"])
    
    N_sa_ = pd.Series(0, index = SA_index)
    N_sa_.update(obs_data[["s","a","m"]].groupby(["s","a"]).count()["m"])
    
    N_sam = pd.Series(0, index = SAM_index)
    N_sam.update(obs_data[["s","a","m","s_"]].groupby(["s","a","m"]).count()["s_"])
    
    N_sams = pd.DataFrame(0, index = SAM_index, columns = S_index)
    N_sams.update(obs_data[["s","a","m","s_","r"]].groupby(["s","a","m","s_"]).count()["r"].unstack(level = -1))
    
    r_sam = pd.Series(0, index = SAM_index)
    r_sam.update(obs_data[["s","a","m","r"]].groupby(["s","a","m"]).sum()["r"])
    
    P_sa = N_sa_.divide(N_s.clip(lower = 1, upper = None), axis = 0, level = 0)
    P_sam = N_sam.divide(N_sa_.clip(lower = 1, upper = None), axis = 0)
    P_sams = N_sams.divide(N_sam.clip(lower = 1, upper = None), axis = 0)
    R_sam = r_sam.divide(N_sam.clip(lower = 1, upper = None), axis =0)
    
    N_sa = pd.Series(0, index = SA_index)
    rep_coef = N_sa.index.get_level_values(1).nunique()
    N_sa.loc[:] = list(N_sam.groupby(level = 0).min().repeat(rep_coef))
    
    inner_R = R_sam.multiply(P_sa, axis = 0).groupby(level = [0,-1]).sum()
    R_sa = P_sam.multiply(inner_R, axis = 0).groupby(level = [0,-1]).sum()
    
    inner_P = P_sams.multiply(P_sa, axis = 0).groupby(level = [0,-1]).sum()
    P_sas = inner_P.multiply(P_sam.reorder_levels(["s","m","a"]), axis = 0).groupby(level = [0,-1]).sum()
    
    return N_sa, R_sa, P_sas