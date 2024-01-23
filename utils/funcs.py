#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re

from tensorflow.keras.models import Sequential
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, LeakyReLU

from sklearn.pipeline import make_pipeline
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler


# In[2]:


def ann(meta, n_first_nodes=10, n_layers=5, activation_func='leaky_relu', activation_func_last='leaky_relu', kernel='normal', 
        lr=1e-4, **kwargs):
    
    model = Sequential()
    
    if activation_func == 'leaky_relu':        
        model.add(Dense(n_first_nodes, input_dim=meta['n_features_in_'], kernel_initializer=kernel))
        model.add(LeakyReLU())
    else:
        model.add(Dense(n_first_nodes, input_dim=meta['n_features_in_'], activation=activation_func, kernel_initializer=kernel))
    
    for i in range(1, n_layers):
        n_nodes = n_first_nodes - int((n_first_nodes - meta['n_outputs_'])/n_layers*i)
        if activation_func == 'leaky_relu':        
            model.add(Dense(n_first_nodes, input_dim=meta['n_features_in_'], kernel_initializer=kernel))
            model.add(LeakyReLU())
        else:
            model.add(Dense(n_first_nodes, input_dim=meta['n_features_in_'], activation=activation_func, kernel_initializer=kernel))
        
    if (activation_func_last is None) or (activation_func_last == 'none'):
        model.add(Dense(meta['n_outputs_'], kernel_initializer=kernel))
    elif activation_func_last == 'leaky_relu':
        model.add(Dense(meta['n_outputs_'], kernel_initializer=kernel))
        model.add(LeakyReLU())
    else:
        model.add(Dense(meta['n_outputs_'], activation=activation_func_last, kernel_initializer=kernel))
    
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    return model


# In[3]:


def build_model(params):
    scaler_type = params.get('scaler', 'std')
    if scaler_type == 'std':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    return make_pipeline(scaler, KerasRegressor(model=ann, random_state=0, verbose=0, **params))


    scaler = StandardScaler()
    return make_pipeline(scaler, KerasRegressor(model=ann, random_state=0, verbose=0, **params))


# In[4]:


def get_random_point(fesb_model, com_step, cutoff):
    while True:
        
        # Composition random generator
        xyz = np.random.rand(3)
        if sum(xyz) > 1.0:
            continue
        if min(xyz) < com_step:
            continue
        if sum(xyz) != 1.0 and (1 - sum(xyz)) < com_step:
            continue
        if fesb_model.predict(xyz.reshape(1, -1)) < cutoff:
            continue
        
        # Morphologies OneHotEncoder random generator
        morph_ohe = np.ones(5)
        morph_ohe[:4] = 0
        np.random.shuffle(morph_ohe)
        
        
        # Pre-stretch random generator
        pre_str = [random.randint(100,301)]
        
        # Thickness random generator
        thick = [random.randint(800,1601)]
                
        point = xyz.tolist() + morph_ohe.tolist() + pre_str + thick
        return point


# In[5]:


def gen_train_data(round_number, da_tag, random_state=0):
    
    def rn_int2str(r: int) -> str:
        return f'{r:02d}'

    def rn_str2int(r: str) -> int:
        return int(r)
    
    round_data = pd.read_csv('data/round_' + round_number + '/train_data/round.'+ round_number + '_train.ohe_data.csv')
    cols = list(round_data.columns)
    
    X_COLS, Y_COLS = cols[1:10], cols[10:]
    
    n_rounds = rn_str2int(round_number)
    random_state=0

    dfs = []

    for i in range(0, n_rounds + 1):
        round_number = rn_int2str(i)
        dfs.append(pd.read_csv('data/round_' + round_number + '/train_data/round.'+ round_number + '_train.ohe_data.csv'))

    df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    magic, n_noisy_replicates, axis = re.split('(\d+)', da_tag)

    assert magic == 'da'

    n_noisy_replicates = int(n_noisy_replicates)
    assert n_noisy_replicates >= 0

    sid_prefix = ''

    x_const_std = 0.01
    y_const_std = 0.05
    sid_len = 8

    if 'x' in axis:
        x_method = 'const_std'
        sid_prefix += 'X'

    if 'y' in axis:
        y_method = 'const_std'
        sid_prefix += 'Y' 

    if y_method == 'const_std':
        std_df = df[Y_COLS] * y_const_std

    Y_STD_COLS = [val + '_std' for val in Y_COLS]
    std_df = std_df.rename(columns={c: s for c, s in zip(Y_COLS, Y_STD_COLS)})

    _sid = df[['sID']].copy()
    std_df = _sid.join(std_df).set_index('sID', drop=True)
    
    noisy_dfs = [df, ]
    rng = np.random.default_rng(random_state)

    for i in range(n_noisy_replicates):
        noisy_x = pd.DataFrame(rng.normal(df[X_COLS], x_const_std), columns=X_COLS)
        noisy_y = pd.DataFrame(rng.normal(df[Y_COLS], std_df), columns=Y_COLS)

        noisy_xy = noisy_x.join(noisy_y)
        noisy_sid = df[['sID']].copy()
        noisy_sid['sID'] += f'-DA_{sid_prefix}{i:05d}'
        noisy_df = noisy_sid.join(noisy_xy).reset_index(drop=True)
        noisy_dfs.append(noisy_df)

#     final_df = pd.concat(noisy_dfs, ignore_index=True).sort_values(by=['sID']).set_index('sID', drop=True)
#     output_path = 'data/round_' + round_number + '/train_data/round.'+ round_number + '_train.data_DA.csv'
#     final_df.to_csv(output_path)
    
    return final_df    


# In[ ]:




