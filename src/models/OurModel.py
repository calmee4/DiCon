import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
import numpy as np
import os
import time
import dill
import torch
import logging
import argparse
import torch.optim as optim
import sys
from copy import deepcopy
import os
import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
from torch.optim import Adam

import dill
import networkx as nx
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
# 导入正常的库
from tqdm import tqdm
sys.path.append("..")
sys.path.append("../..")
import torch.nn.functional as F
from prettytable import PrettyTable
from collections import defaultdict
from rdkit import Chem
from torch_geometric.nn import SignedGCN
from .gnn import GNNGraph
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR, StepLR
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from utils.layers import GraphConvolution, Fastformer

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import random

class SGCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,dropout=0.2):
        super(SGCNModel, self).__init__()
        self.signed_gcn = SignedGCN(in_channels, hidden_channels, num_layers)

        # LayerNorm + Dropout
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Projection head for downstream task
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Learnable gate for pretrain embedding fusion
        self.gate = nn.Parameter(torch.tensor(0.5))
    def forward(self, pretrained_emb, pos_edge_index, neg_edge_index):
        x = pretrained_emb
        for i, conv in enumerate(self.signed_gcn.convs):
            x = conv(x, pos_edge_index, neg_edge_index)
            x = F.relu(x)
            x = self.norms[i](x)
            x = self.dropouts[i](x)
        # projection head
        x = self.proj_head(x)
        # 融合 pretrain embedding
        return self.gate * x + (1 - self.gate) * pretrained_emb


class ResidualGRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first, dropout=0, bidirectional=False):
        super().__init__()
        self.bi = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, batch_first=batch_first,
                          bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        # 残差前先把维度对齐
        self.proj = nn.Linear(out_dim, input_size) if out_dim != input_size else nn.Identity()
        self.ln = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        # 学习一个门控，控制残差强度
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, h0=None):
        out, h = self.gru(x, h0)         # [B,T,out_dim]
        out = self.proj(out)             # 对齐到 input_size 以便残差
        out = self.dropout(out)
        out = x + self.beta * out        # Gated residual
        out = self.ln(out)
        return out, h


class PositionalEncoding(nn.Module): #用于消融实验的Positional Encoder
    """
    Standard Positional Encoding for Transformer models.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # The input x is batch-first, but pe is [seq_len, 1, dim]. We need to transpose x.
        x = x.transpose(0, 1) # [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1) # back to [batch_size, seq_len, embedding_dim]

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接 + LayerNorm
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)
def sample_positive_ddi(tensor_ddi_adj_positive, sample_ratio):
    """
    对正向DDI矩阵进行采样

    Args:
        tensor_ddi_adj_positive: 正向DDI邻接矩阵
        sample_ratio: 采样比例 (0.1 - 1.0)

    Returns:
        采样后的正向DDI矩阵
    """
    if sample_ratio >= 1.0:
        return tensor_ddi_adj_positive

    print(f"Sampling Positive DDI with ratio: {sample_ratio}")

    # 步骤1: 获取所有正向DDI边，去重（只取上三角）
    pos_edges = torch.triu(tensor_ddi_adj_positive > 0, diagonal=1).nonzero()
    total_edges = pos_edges.shape[0]

    # 步骤2: 随机采样
    keep_size = int(total_edges * sample_ratio)
    keep_indices = torch.randperm(total_edges)[:keep_size]
    sampled_edges = pos_edges[keep_indices]

    # 步骤3: 构建新的DDI矩阵
    pos_ddi_sampled = torch.zeros_like(tensor_ddi_adj_positive)

    # 填充采样后的边（保持对称性）
    for edge in sampled_edges:
        i, j = edge[0].item(), edge[1].item()
        pos_ddi_sampled[i, j] = tensor_ddi_adj_positive[i, j]
        pos_ddi_sampled[j, i] = tensor_ddi_adj_positive[j, i]  # 对称

    print(f"Total edges: {total_edges}, Sampled edges: {keep_size}")
    return pos_ddi_sampled


class OurModel(torch.nn.Module):
    def __init__(
            self,
            args,
            tensor_ddi_adj,
            tensor_ddi_adj_positive,
            emb_dim,
            voc_size,
            dropout,
            pretrained_embeddings,
            sample=1,
            device=torch.device('cpu')
    ):
        super(OurModel, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        # Embedding of all entities
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),  # 疾病
            torch.nn.Embedding(voc_size[1], emb_dim),  # 手术
            torch.nn.Embedding(voc_size[2], emb_dim),  # 药物embedding
        ])

        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        '''
        Signed Network
        '''
        self.SGCNModel = SGCNModel(emb_dim, emb_dim)
        self.pos_ddi = sample_positive_ddi(tensor_ddi_adj_positive, sample)
        self.neg_ddi = tensor_ddi_adj
        '''
        GRU/LSTM/GRU-Residual
        '''
        # #  GRU
        # self.seq_encoders = torch.nn.ModuleList([
        #     torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
        #     torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
        #     torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        # ])
        # # LSTM
        # self.seq_encoders = nn.ModuleList([
        #     nn.LSTM(emb_dim, emb_dim, batch_first=True),
        #     nn.LSTM(emb_dim, emb_dim, batch_first=True),
        #     nn.LSTM(emb_dim, emb_dim, batch_first=True)
        # ])
        #Transformer+Positional
        # encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=emb_dim * 2,
        #                                            dropout=dropout, batch_first=True)
        # self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        # self.seq_encoders = nn.ModuleList([
        #     nn.TransformerEncoder(encoder_layer, num_layers=2),
        #     nn.TransformerEncoder(encoder_layer, num_layers=2),
        #     nn.TransformerEncoder(encoder_layer, num_layers=2)
        # ])
        # # Residual-GRU (OurModel)
        self.seq_encoders = nn.ModuleList([
            ResidualGRU(emb_dim, emb_dim, batch_first=True),
            ResidualGRU(emb_dim, emb_dim, batch_first=True),
            ResidualGRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.tensor_ddi_adj = tensor_ddi_adj
        self.tensor_ddi_adj_positive = tensor_ddi_adj_positive
        self.init_weights()
        self.attn_q_proj = torch.nn.Linear(6 * emb_dim, emb_dim)  # Q from patient_repr
        self.attn_k_proj = torch.nn.Linear(emb_dim, emb_dim)  # K from molecule embeddings
        self.attn_v_proj = torch.nn.Linear(emb_dim, emb_dim)  # V from molecule embeddings
        self.drug_noise = nn.Parameter(torch.randn(voc_size[2], emb_dim))
        self.m = nn.Parameter(torch.tensor(0.5))
        self.device = device
        self.diag_hidden_norm = nn.LayerNorm(emb_dim)
        self.proc_hidden_norm = nn.LayerNorm(emb_dim)
        self.med_hidden_norm = nn.LayerNorm(emb_dim)
        hidden_dim = self.emb_dim*2
        score_extractor = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout=args.dp),  # ✅ 只保留一个 Residual
            nn.Linear(hidden_dim, 1)
        )
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(p=args.dp)
        # --- 诊断嵌入层 --- 消融实验2 就是把下 面pretrained注释掉
        self.emb_diag = nn.Embedding(voc_size[0], args.dim)
        if pretrained_embeddings and 'diag' in pretrained_embeddings:
            print("Initializing 'diag' embeddings with pre-trained weights.")
            self.emb_diag.weight.data.copy_(pretrained_embeddings['diag'])
        self.emb_proc = nn.Embedding(voc_size[1], args.dim)
        if pretrained_embeddings and 'proc' in pretrained_embeddings:
            print("Initializing 'proc' embeddings with pre-trained weights.")
            self.emb_proc.weight.data.copy_(pretrained_embeddings['proc'])
        self.emb_med = nn.Embedding(voc_size[2], args.dim)
        if pretrained_embeddings and 'med' in pretrained_embeddings:
            print("Initializing 'med' embeddings with pre-trained weights.")
            self.emb_med.weight.data.copy_(pretrained_embeddings['med'])
        self.pos_ddi_index = (self.pos_ddi > 0).nonzero().t().contiguous().long().to(self.device)
        self.neg_ddi_index = (self.neg_ddi > 0).nonzero().t().contiguous().long().to(self.device)
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
    def forward(self, patient_data):
        seq_diag, seq_proc, seq_med = [], [], []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm_id, adm in enumerate(patient_data):
            i1 = mean_embedding(
                self.dropout(self.emb_diag(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i2 = mean_embedding(
                self.dropout(self.emb_proc(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            if adm == patient_data[0]:
                i3 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                i3 = mean_embedding(
                    self.dropout(self.emb_proc(torch.LongTensor(adm_last[2]).unsqueeze(dim=0).to(self.device))))
            seq_diag.append(i1)
            seq_proc.append(i2)
            seq_med.append(i3)
        seq_diag = torch.cat(seq_diag, dim=1)
        seq_proc = torch.cat(seq_proc, dim=1)
        seq_med = torch.cat(seq_med, dim=1)


        output_diag,  hidden_diag = self.seq_encoders[0](seq_diag)
        output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
        output_med, hidden_med = self.seq_encoders[2](seq_med)

        hidden_diag = self.diag_hidden_norm(hidden_diag)
        hidden_proc = self.proc_hidden_norm(hidden_proc)
        hidden_med = self.med_hidden_norm(hidden_med)
        seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med], dim=-1)
        last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1]], dim=-1)
        # Ablation 1  只用当前一次的 不建模时序
        # current_adm = patient_data[-1]
        #
        # # 1. 获取当前就诊的诊断嵌入
        # current_diag_codes = torch.LongTensor(current_adm[0]).unsqueeze(dim=0).to(self.device)
        # current_diag_repr = mean_embedding(self.dropout(self.emb_diag(current_diag_codes)))
        #
        # # 2. 获取当前就诊的手术嵌入
        # current_proc_codes = torch.LongTensor(current_adm[1]).unsqueeze(dim=0).to(self.device)
        # current_proc_repr = mean_embedding(self.dropout(self.emb_proc(current_proc_codes)))
        #
        # # 3. 获取上一次就诊的用药嵌入 (作为历史信息)
        # if len(patient_data) > 1:  # 如果存在历史就诊
        #     last_adm = patient_data[-2]
        #     last_med_codes = torch.LongTensor(last_adm[2]).unsqueeze(dim=0).to(self.device)
        #     # 确保即使上次用药为空列表，也能正常工作
        #     if last_med_codes.shape[1] > 0:
        #         prev_med_repr = mean_embedding(self.dropout(self.emb_med(last_med_codes)))
        #     else:
        #         prev_med_repr = torch.zeros(1, self.emb_dim).to(self.device)
        # else:  # 如果这是第一次就诊，没有历史用药记录
        #     prev_med_repr = torch.zeros(1, self.emb_dim).to(self.device)
        #
        # # 应用LayerNorm
        # current_diag_repr = self.diag_hidden_norm(current_diag_repr)
        # current_proc_repr = self.proc_hidden_norm(current_proc_repr)
        # prev_med_repr = self.med_hidden_norm(prev_med_repr)
        #
        # # 4. 构建患者表征 patient_repr
        # # 将三个表征拼接成一个向量
        # patient_repr = torch.cat([current_diag_repr.flatten(),
        #                           current_proc_repr.flatten(),
        #                           prev_med_repr.flatten()])
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
        # without SGCN/A+/A-
        # without_ddi_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        # drug_emb_SGCN = self.SGCNModel(self.emb_med.weight, self.pos_ddi_index, without_ddi_index)  # Ablation 2
        # drug_emb_SGCN = self.SGCNModel(self.emb_med.weight, without_ddi_index, self.neg_ddi_index)
        drug_emb_SGCN = self.SGCNModel(self.emb_med.weight, self.pos_ddi_index, self.neg_ddi_index)
        q = self.attn_q_proj(patient_repr)  # [emb_size]       Query
        k = self.attn_k_proj(drug_emb_SGCN)  # [med_size,emb_size]   Key
        v = self.attn_v_proj(drug_emb_SGCN)  # [med_size,emb_size]   Value
        raw_scores = torch.matmul(k, q) / math.sqrt(self.emb_dim)  # [med_size]
        # drug_feat = dru       #消融实验1 without Signed Network
        drug_feat = drug_emb_SGCN + v
        mlp_scores = self.score_extractor(drug_feat).squeeze(1) # [med_size]
        drug_scores = self.alpha*raw_scores+(1-self.alpha)*mlp_scores
        score = drug_scores.unsqueeze(0)
        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        batch_neg_positive = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj_positive).sum()
        return score, batch_neg,batch_neg_positive