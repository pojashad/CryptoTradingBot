#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pdb

filters= [128,64]
batch_sizes = [8*3,16*3] #The batch size will be split in 3, one part for each train partition
kernel_size = [128,64]
dilation_rate = [3,5,16]
num_res_blocks = [0,1,2]
lookback_window = [24*7,24*14,24*28]
forecast_window = [6,12,24]
test_partitions = [0,1,2,3,4]

combos = np.zeros((len(filters)*len(batch_sizes)*len(kernel_size)*len(dilation_rate)*len(num_res_blocks)*len(lookback_window)*len(forecast_window)*len(test_partitions),8))
i=0
for f in filters:
    for s in batch_sizes:
        for k in kernel_size:
            for dr in dilation_rate:
                for n in num_res_blocks:
                    for lb in lookback_window:
                        for fw in forecast_window:
                            for tp in test_partitions:
                                combos[i]=[f,s,k,dr,n,lb,fw,tp]
                                i+=1

combo_df = pd.DataFrame()
combo_df['filters']=combos[:,0]
combo_df['batch_size']=combos[:,1]
combo_df['kernel_size']=combos[:,2]
combo_df['dilation_rate']=combos[:,3]
combo_df['num_res_blocks']=combos[:,4]
combo_df['lookback_window']=combos[:,5]
combo_df['forecast_window']=combos[:,6]
combo_df['test_partition']=combos[:,7]
combo_df.to_csv('param_combos.csv')
