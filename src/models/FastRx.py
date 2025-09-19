import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.layers import GraphConvolution,Fastformer

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class FastRx(nn.Module):
    def __init__(self, args, vocab_size, ehr_adj, ddi_adj, emb_dim=256):
        super(FastRx, self).__init__()
        device = torch.device('cuda:{}'.format(args.cuda))
        self.device = torch.device('cuda:{}'.format(args.cuda))
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.emb_dim_ff = 128
        self.fastformer = Fastformer(dim = 2*self.emb_dim_ff, decode_dim = self.emb_dim)
        self.dropout = nn.Dropout(p=0.2)

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, self.emb_dim_ff)

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

    def forward(self, input):
	    # patient health representation
        i1_seq, i2_seq = [], []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i2 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))

            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        i1_seq = self.cnn1d(i1_seq.permute(1, 0, 2))
        i2_seq = self.cnn1d(i2_seq.permute(1, 0, 2))
        i1_seq = i1_seq.permute(1, 0, 2)
        i2_seq = i2_seq.permute(1, 0, 2)

        h = torch.cat([i1_seq, i2_seq], dim=-1) # (seq, dim*2)

        mask = torch.ones(1, self.emb_dim).to(torch.bool).to(self.device)

        feat = self.fastformer(h, mask).squeeze(0)

        # graph memory module
        '''I:generate current input'''
        query = feat[-1:] # (1,dim)
        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)

        if len(input) > 1:
            history_keys = feat[:(feat.size(0)-1)] # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        # print(query.shape, drug_memory.t().shape)
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        result = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(result)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
            return result, batch_neg
        else:
            return result
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, HeteroGraph, device):
        super().__init__()
        self.device = device  # 保存设备信息
        self.HeteroGraph = HeteroGraph
        self.x_dict = HeteroGraph.x_dict
        self.edge_index_dict = HeteroGraph.edge_index_dict
        self.hetero_conv = HeteroConv({
            ('node', 'pos', 'node'): SAGEConv((-1, -1), hidden_channels),
            ('node', 'neg', 'node'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')  # 异构图的图卷积

        self.to(self.device)

    def forward(self):
        # 确保输入数据也在正确设备上
        x_dict = {k: v.to(self.device) for k, v in self.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in self.edge_index_dict.items()}
        x_dict = self.hetero_conv(self.x_dict, self.edge_index_dict)
        node_emb = x_dict['node']  # shape: [num_nodes, hidden_dim]
        Hetero_feature = node_emb.mean(dim=0).unsqueeze(0)
        return Hetero_feature

class FastRx1(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj,ddi_adj_positive, HeteroGraph_data,emb_dim=256, device=torch.device('cpu:0')):
        super(FastRx1, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.emb_dim_ff = 128
        self.fastformer = Fastformer(dim = 2*self.emb_dim_ff, decode_dim = self.emb_dim)
        self.dropout = nn.Dropout(p=0.2)

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, self.emb_dim_ff)
        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.tensor_ddi_adj_positive = torch.FloatTensor(ddi_adj_positive).to(device)
        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )



        self.output = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )
        self.HeterGNN = HeteroGNN(emb_dim, HeteroGraph=HeteroGraph_data, device=self.device)

    def forward(self, input):
	    # patient health representation
        i1_seq, i2_seq = [], []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i2 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))

            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        i1_seq = self.cnn1d(i1_seq.permute(1, 0, 2))
        i2_seq = self.cnn1d(i2_seq.permute(1, 0, 2))
        i1_seq = i1_seq.permute(1, 0, 2)
        i2_seq = i2_seq.permute(1, 0, 2)

        h = torch.cat([i1_seq, i2_seq], dim=-1) # (seq, dim*2)

        mask = torch.ones(1, self.emb_dim).to(torch.bool).to(self.device)
        feat = self.fastformer(h, mask).squeeze(0)
        # graph memory module
        '''I:generate current input'''
        query = feat[-1:] # (1,dim)
        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        # drug_memory = self.HeterGNN()
        drug_memory = drug_memory.repeat(112, 1)
        if len(input) > 1:
            history_keys = feat[:(feat.size(0)-1)] # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        # print(query.shape, drug_memory.t().shape)

        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        result = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        batch_pos = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj_positive).sum()
        return result, batch_neg,batch_pos