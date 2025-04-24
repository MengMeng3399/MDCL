import sys
sys.argv = ['run.py']

import itertools
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
from torch.nn.parameter import Parameter
import numpy as np
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
from utils.preprocess import *
import dgl
import dgl.data
import json

from torch.nn.utils.rnn import pad_sequence


from torch.nn.utils.rnn import pad_sequence

train_file='data/dataset/HGA/train.csv'
train = pd.read_csv(train_file)

warm_test_file='data/dataset/HGA/warm_test.csv'
warm_test = pd.read_csv(warm_test_file)

cold_test_file='data/dataset/HGA/cold_test.csv'
cold_test = pd.read_csv(cold_test_file)

all_test_file='data/dataset/HGA/all_test.csv'
all_test = pd.read_csv(all_test_file)

invoke_data_file ="data/HGA/Invoke_edge.csv"
invoke = pd.read_csv(invoke_data_file)

api_content = torch.load("Ddata/HGA/newf.pt")


class Recommender(nn.Module):

    def __init__(self,n_mashup,n_api,api_content_feature_size,content_feature,warm_api,cold_api,adj_matrix,num_layers=3):
        super(Recommender, self).__init__()
        """
        Args:
            n_mashup: Mashup数量
            n_api: API数量
            api_content_feature_size: API内容特征的维度
            content_feature: API内容特征数据
            adj_matrix: n_api)
            num_layers: LightGCN的传播层数
        """
        # args_config.dim = 64
        self.emb_size = 64
        self.n_mashup = n_mashup
        self.n_api = n_api
        self.api_content_feature_size = api_content_feature_size
        self.content_feature = content_feature
        self.num_layers = num_layers
        self.cold_api = cold_api
        self.warm_api = warm_api

     
        self.adj_matrix = adj_matrix
    
        self.fc = nn.Sequential(
                nn.Linear(self.api_content_feature_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                ) 
        
    
        p_weight = np.random.randn(self.n_mashup, self.emb_size) * 0.01  
        q_weight = np.random.randn(self.n_api, self.emb_size) * 0.01

        # mashup的嵌入矩阵
        self.mashup_emb = torch.nn.Embedding(self.n_mashup, self.emb_size)
        self.mashup_emb.weight.data.copy_(torch.tensor(p_weight))
        self.mashup_emb.weight.requires_grad = True

        # api的嵌入矩阵
        self.api_emb = torch.nn.Embedding(self.n_api, self.emb_size)
        self.api_emb.weight.data.copy_(torch.tensor(q_weight))
        self.api_emb.weight.requires_grad = True

    def lightgcn_propagation(self):
  
        all_embeddings = torch.cat([self.mashup_emb.weight, self.api_emb.weight], dim=0)  # (n_mashup + n_api, emb_size)

        embeddings_list = [all_embeddings]


        row_sum = self.adj_matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj_matrix = d_mat_inv_sqrt @ self.adj_matrix @ d_mat_inv_sqrt

        for layer in range(self.num_layers):
            all_embeddings = torch.matmul(norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)


        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)

    
        mashup_emb_final, api_emb_final = torch.split(final_embeddings, [self.n_mashup, self.n_api], dim=0)
        return mashup_emb_final, api_emb_final

    def forward(self):
        
      
        mashup_emb_final, api_emb_final = self.lightgcn_propagation()
      
        self.content_emb = self.fc(self.content_feature)
        
        # cold_api_indices = torch.tensor(list(self.cold_api), dtype=torch.long, device=api_emb_final.device)

        # updated_api_emb_final = api_emb_final.clone()

        # updated_api_emb_final[cold_api_indices] = self.content_emb[cold_api_indices]

        # return updated_api_emb_final, mashup_emb_final

        return self.content_emb, mashup_emb_final
    
    
    def compute_contrastive_loss(self,data,neg_samples_api, neg_samples_mashup,alpha,beta,a,b,c):


        loss1 = self.recom_level_loss(data,beta,neg_samples_api, neg_samples_mashup)
        # loss1 =self.bpr_loss(data, neg_samples_api)
        loss2 = self.api_level_loss(data,alpha,neg_samples_api)
        loss3 =self.group_level_loss(data,neg_samples_api)
        # return loss1

        reg_loss = self.regs()
        # loss3=0.0
      

        return (1-a)*loss1+ a*loss2 + b*loss3 + c*reg_loss, loss1,loss2,loss3

    
    def api_level_loss(self, data, alpha, neg_samples_api, temperature=0.1):
    
        api_indices = torch.tensor(
            [api for api in data['Api_ID'].unique() if api in self.warm_api],
            dtype=torch.long, device=self.content_emb.device)

        if len(api_indices) == 0:
            return 0.0  # 无有效样本

        # 正样本嵌入
        positive_cf_embed = self.api_emb(api_indices)
        positive_content_embed = self.content_emb[api_indices]

        # 负样本嵌入
        negative_api_indices = torch.stack([neg_samples_api[api.item()] for api in api_indices])  # (batch_size, num_neg_samples)
        negative_cf_embed = self.api_emb(negative_api_indices)          # (batch_size, num_neg_samples, embedding_dim)
        negative_content_embed = self.content_emb[negative_api_indices] # (batch_size, num_neg_samples, embedding_dim)

        # 正样本相似度
        pos_sim = torch.exp(torch.sum(positive_cf_embed * positive_content_embed, dim=1) / temperature)

        # 负样本相似度
        neg_sim_1 = torch.exp(torch.sum(positive_cf_embed.unsqueeze(1) * negative_content_embed, dim=2)).sum(dim=1)
        neg_sim_2 = torch.exp(torch.sum(positive_content_embed.unsqueeze(1) * negative_cf_embed, dim=2)).sum(dim=1)

        # Clamp values to prevent numerical instability
        pos_sim = torch.clamp(pos_sim, min=1e-10)
        neg_sim_1 = torch.clamp(neg_sim_1, min=1e-10)
        neg_sim_2 = torch.clamp(neg_sim_2, min=1e-10)

        # 计算对比损失
        loss_1 = -torch.log(pos_sim / (pos_sim + neg_sim_1))
        loss_2 = -torch.log(pos_sim / (pos_sim + neg_sim_2))

        return (alpha * loss_1 + (1 - alpha) * loss_2).mean()
    
    def recom_level_loss(self, data, beta, neg_samples_api, neg_samples_mashup, temperature=0.1):
    
        # 获取 Mashup 和 API 索引
        mashup_indices = torch.tensor(data['Mashup_ID'], dtype=torch.long, device=self.mashup_emb.weight.device)
        api_indices = torch.tensor(data['Api_ID'], dtype=torch.long, device=self.api_emb.weight.device)

        # 正样本嵌入
        mashup_embed = self.mashup_emb(mashup_indices)  # Mashup 嵌入
        api_embed = self.api_emb(api_indices)          # API 嵌入

        # 负样本嵌入
        negative_api_indices = torch.cat([neg_samples_api[api.item()].unsqueeze(0) for api in api_indices])
        negative_mashup_indices = torch.cat([neg_samples_mashup[mashup.item()].unsqueeze(0) for mashup in mashup_indices])

        negative_api_embed = self.api_emb(negative_api_indices)
        negative_mashup_embed = self.mashup_emb(negative_mashup_indices)

        # 正样本相似度 (batch_size)
        pos_sim_ui = torch.sum(mashup_embed * api_embed, dim=1) / temperature
        pos_sim_iu = torch.sum(api_embed * mashup_embed, dim=1) / temperature

        # 负样本相似度 (batch_size, K)
        neg_sim_ui = torch.sum(mashup_embed.unsqueeze(1) * negative_api_embed, dim=2) / temperature  # (batch_size, num_neg_samples)
        neg_sim_iu = torch.sum(api_embed.unsqueeze(1) * negative_mashup_embed, dim=2) / temperature  # (batch_size, num_neg_samples)

        # Clamp values to prevent numerical instability
        pos_sim_ui = torch.clamp(pos_sim_ui, min=1e-10)
        pos_sim_iu = torch.clamp(pos_sim_iu, min=1e-10)
        neg_sim_ui = torch.clamp(neg_sim_ui, min=1e-10)
        neg_sim_iu = torch.clamp(neg_sim_iu, min=1e-10)

        # Log-Sum-Exp 提升稳定性
        loss_ui = -torch.log(torch.exp(pos_sim_ui) / (torch.exp(pos_sim_ui) + torch.exp(neg_sim_ui).sum(dim=1)))
        loss_iu = -torch.log(torch.exp(pos_sim_iu) / (torch.exp(pos_sim_iu) + torch.exp(neg_sim_iu).sum(dim=1)))

        # 加权求和
        total_loss = beta * loss_ui + (1 - beta) * loss_iu
        return total_loss.mean()
    
    def group_level_loss(self, data, neg_samples_api, temperature=0.1):
     
        mashup_indices = torch.tensor(data['Mashup_ID'].unique(), dtype=torch.long, device=self.content_emb.device)
        total_loss = 0.0

        for mashup_id in mashup_indices:
            # 获取当前 mashup 的 API 集合
            api_indices = torch.tensor(
                data[data['Mashup_ID'] == mashup_id.item()]['Api_ID'].tolist(),
                dtype=torch.long, device=self.content_emb.device)

            if len(api_indices) <= 1:
                continue

            # 随机选取一个目标 API，并获取剩余 API
            target_api_index = api_indices[torch.randint(len(api_indices), (1,))]
            remaining_api_indices = api_indices[api_indices != target_api_index]

            # 计算加权偏好嵌入
            remaining_embeddings = self.api_emb(remaining_api_indices)
            user_embedding = self.mashup_emb(mashup_id)
            cosine_sim = F.cosine_similarity(user_embedding.unsqueeze(0), remaining_embeddings, dim=1)
            softmax_weights = F.softmax(cosine_sim / temperature, dim=0)  # 温度平滑

            preference_embedding = torch.sum(softmax_weights.unsqueeze(1) * remaining_embeddings, dim=0)

            # 正样本与负样本相似度
            target_embedding = self.content_emb[target_api_index]
            negative_api_indices = neg_samples_api[target_api_index.item()]
            negative_embeddings = self.content_emb[negative_api_indices]

            pos_sim = torch.exp(torch.sum(preference_embedding * target_embedding) / temperature)
            neg_sim = torch.exp(torch.matmul(preference_embedding, negative_embeddings.T) / temperature).sum()

            # Log-Sum-Exp 技巧
            loss = -torch.log(pos_sim / (pos_sim + neg_sim))
            total_loss += loss

        return total_loss / len(mashup_indices) if len(mashup_indices) > 0 else 0.0
    
    def regs(self):
        """
        计算模型参数的正则化损失。
        
        Args:
            lambda_reg (float): 正则化系数，控制正则化的强度。
        
        Returns:
            torch.Tensor: 正则化损失。
        """
        lambda_reg=1e-6
        # 遍历模型的所有参数，排除不需要正则化的参数（如偏置项 bias）
        reg_loss = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:  # 过滤掉 bias 参数
                reg_loss += torch.norm(param, p=2) ** 2  # 使用 L2 正则化

        return lambda_reg * reg_loss
    
    def bpr_loss(self, data, neg_samples_api, temperature=0.1):
        """
        计算单侧 BPR 损失，仅针对用户到物品 (User-to-Item) 的排序优化。
        """
        # 获取 Mashup 和 API 索引
        mashup_indices = torch.tensor(data['Mashup_ID'], dtype=torch.long, device=self.mashup_emb.weight.device)
        api_indices = torch.tensor(data['Api_ID'], dtype=torch.long, device=self.api_emb.weight.device)

        # 正样本嵌入
        mashup_embed = self.mashup_emb(mashup_indices)  # Mashup 嵌入
        api_embed = self.api_emb(api_indices)          # API 嵌入

        negative_api_indices = []
        for api in api_indices:
            neg_sample = random.choice(neg_samples_api[api.item()])  # 从负样本池中随机选择一个负样本
            negative_api_indices.append(neg_sample)
        
        negative_api_indices = torch.tensor(negative_api_indices, dtype=torch.long, device=self.api_emb.weight.device)
        negative_api_embed = self.api_emb(negative_api_indices)  # 负样本嵌入

        # 计算正样本评分
        pos_sim_ui = torch.sum(mashup_embed * api_embed, dim=1) / temperature  # 正样本相似度

        # 计算负样本评分
        neg_sim_ui = torch.sum(mashup_embed * negative_api_embed, dim=1) / temperature  # 负样本相似度

        # 计算BPR损失（正负样本评分差）
        loss_ui = -torch.log(torch.sigmoid(pos_sim_ui - neg_sim_ui))  # User -> Item 的损失

        return loss_ui.mean()
            


model = Recommender(num_mashup,num_api,apicontent_size,api_content,warm_api,cold_api,norm_adj_matrix)# .to(device)

optimizer = torch.optim.Adam(
    itertools.chain(model.parameters()), lr=2e-3
)


import datetime
import os


class TeeOutput:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保内容立即写入

    def flush(self):
        for f in self.files:
            f.flush()

def log_to_file_with_terminal_output(log_dir=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
    
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"log_{current_time}.txt"

            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                log_filepath = os.path.join(log_dir, log_filename)
            else:
                log_filepath = log_filename

            # 打开日志文件
            with open(log_filepath, "w") as log_file:
                tee = TeeOutput(sys.stdout, log_file)

                # 替换sys.stdout
                original_stdout = sys.stdout
                sys.stdout = tee

                try:
                    # 执行传入的函数
                    func(*args, **kwargs)
                finally:
                    # 恢复sys.stdout
                    sys.stdout = original_stdout
                    print(f"Logs saved to {log_filepath}")

        return wrapper
    return decorator

all_logits = []

@log_to_file_with_terminal_output(log_dir="D:/reading_paper_P3/1A_experimental_data/log/final")
# def main(neg_api=neg_samples_api,neg_mashup=neg_samples_mashup,k=256,i=0.2,j=0.2):
def main(a_alpha=0.5,a_beta=0.5):
    for e in range(60):
        
        try:
            # forward
            api_emb, mashup_emb = model()
            loss, loss1, loss2, loss3 = model.compute_contrastive_loss(
                train,
                neg_samples_api,
                neg_samples_mashup,
                alpha=a_alpha,
                beta=a_beta,
                a=0.2,
                b=0.2,
                c=0.01
            )

            # backward
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if e % 5 == 0:
                print(f"In epoch {e}, loss: {loss}, loss1: {loss1}, loss2: {loss2}, loss3: {loss3}")
        
        except ValueError as ve:
            print(f"ValueError encountered in epoch {e}: {ve}")
            continue  # Skip this epoch and proceed to the next one
        except Exception as ex:
            print(f"Unexpected error encountered in epoch {e}: {ex}")
            continue  # Skip this epoch and proceed to the next one


    # ----------- check results ------------------------ 

    with torch.no_grad():
        try:
            # warm:
            warm_metrics = evaluate(mashup_emb, api_emb, train, warm_test, warm_api, k=10)
            # warm_metrics_2 = evaluate(mashup_emb, api_emb, train, warm_test, warm_api, k=30)
            # cold:
            cold_metrics = evaluate(mashup_emb, api_emb, train, cold_test, cold_api, k=10)
            # cold_metrics_2 = evaluate(mashup_emb, api_emb, train, cold_test, cold_api, k=30)
            #
            all_metrics = evaluate(mashup_emb, api_emb, train, all_test, warm_api.union(cold_api), k=10)
            # all_metrics_2 = evaluate(mashup_emb, api_emb, train, all_test, warm_api.union(cold_api), k=30)

        except Exception as e:
            print(f"Error encountered during evaluation: {e}")
            warm_metrics, cold_metrics, all_metrics = None, None, None

        
    torch.cuda.empty_cache()

    print("alpha = {},beta = {},top = {}:".format(a_alpha,a_beta,10),warm_metrics,cold_metrics,all_metrics) 
    # print("top = {}:".format(30),warm_metrics_2,cold_metrics_2,all_metrics_2) 