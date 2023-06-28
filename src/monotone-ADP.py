"""
Algorithm 1. off-line pre-learning process of monotone-ADP(small size IETS)

Small size IETS is with 6 electricity buses and 5 thermal nodes, 1 BS, and 1 TS.

"""
import numpy as np
# Step1: generate a set of training samples Ω,containing trajectories of exogenous information
# all parameters of small size IETS
# TDN,CHP,EDN,TS,ambient temperature,expectation of real-time price
# ???
# parameters of RES
RES_UPPER_ACTIVE_POWER_OUTPUT = [
    0,0,0,0,0,0,
    0,107.38,39.51,83.28,128.02,156.90,
    145.77,169.98,165.62,130.25,-89.93,-42.51,
    -15.90,0,0,0,0,0
] #kW
# parameters of BS
BS_MAX_CHARGING_POWER = 24 #kW
BS_MAX_DISCHARGING_POWER = 24 #kW
BS_UPPER_LIMIT_ENERGY_=80 #kWH
BS_LOWER_LIMIT_ENERGY = 0 #kWH
BS_CHARGING_EFFICIENCY = 0.98
BS_DISCHARGING_EFFICIENCY = 0.98
BS_OR_COST_COEF=0.01 # $/kW


traing_samples = np.random.rand(1000)
T = 24
N = 2500
V = np.zeros((T,N))
for n in range(1, N):
    omega = traing_samples[0]
    for t in range(1, T):
        vtn=min(C)

