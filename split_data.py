import sys
sys.argv = ['run.py']
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
import logging
from utils.parser import parse_args
import dgl
import dgl.data
import pandas as pd
from utils.preprocess import *
import json

import logging

data_file='data/invoke_edge.csv'
invoke_data = pd.read_csv(data_file)

#划分数据集： 
ratio=0.2  
train,warm_test,cold_test=data_split(invoke_data,ratio)
all_test = pd.concat([warm_test, cold_test], ignore_index=True)

##测试：
unique_items = (train['Mashup_ID'].unique()).tolist()
warm = (warm_test['Mashup_ID'].unique()).tolist()
cold = (cold_test['Mashup_ID'].unique()).tolist()

warm_list=[]
for i in warm:
    if i not in unique_items:
        warm_list.append(i)
        
cold_list=[]
for i in cold:
    if i not in unique_items:
        cold_list.append(i)
print(warm_list,cold_list)


def encode_sample(sample_dict):
    feature_tensors = [torch.tensor(value, dtype=torch.float32).flatten() for value in sample_dict.values()]
    combined_vector = torch.cat(feature_tensors, dim=0)
    # print(combined_vector.shape)
    return combined_vector


api_conten_file ='D:/mashup_project/my-DeepCTR/examples/mashup-api/paper3/data/HGA/api_em.json'

with open(api_conten_file, 'r') as fd:
        api_content_data = fd.read()
        Apicontent = json.loads(api_content_data)

encoded_data = [encode_sample(sample) for sample in Apicontent]
api_content = torch.stack(encoded_data)


def data_split(df,ratio):
    '''
    input: dataframe 的 df  三列[Mashup_ID,交互关系,Api_ID]
            ratio: 挑选冷启动API的比例
    outout: cold_test dataframe,  train dataframe, warm_test dataframe.
    '''

    unique_items = df['Api_ID'].unique()
    unique_users = df['Mashup_ID'].unique()

    # Step 1: 随机挑选20%的item作为冷启动item，确保每个用户至少有一个非冷启动item
    np.random.seed(42) 
    cold_start_items = set()
    while True:
        candidate_items = set(np.random.choice(unique_items, size=int(len(unique_items) * ratio), replace=False))
        user_item_map = df.groupby('Mashup_ID')['Api_ID'].apply(set)
        if all(len(items - candidate_items) > 0 for items in user_item_map):
            cold_start_items = candidate_items
            break

    non_cold_items = set(unique_items) - cold_start_items

    non_cold_df = df[df['Api_ID'].isin(non_cold_items)]

    train_indices = []
    test_indices = []
    train_apis = set() 

    for Mashup_ID, group in non_cold_df.groupby('Mashup_ID'):
        indices = group.index.values
        np.random.shuffle(indices)
        split = int(len(indices) * 0.2)
        test_group_indices = indices[:split]
        test_indices.extend(test_group_indices)
        remaining_group_indices = indices[split:]
        remain_apis = set(df.loc[remaining_group_indices, 'Api_ID'])
        for idx in test_group_indices:
            api_id = df.loc[idx, 'Api_ID']
            if api_id not in train_apis and api_id not in remain_apis:
                # 如果Api不在训练集中，将一条对应的记录移到训练集
                train_indices.append(idx)
                train_apis.add(api_id)
                test_indices.remove(idx)  # 从测试集移除

    
        train_indices.extend(remaining_group_indices)
        train_apis.update(df.loc[remaining_group_indices, 'Api_ID'])
    train = df.loc[train_indices]
    warm_test = df.loc[test_indices]
    cold_test = df[df['Api_ID'].isin(cold_start_items)]
    warm_test = add_negative_samples(warm_test, df, non_cold_items)
    cold_test = add_negative_samples(cold_test, df, cold_start_items)


    return train,warm_test,cold_test
