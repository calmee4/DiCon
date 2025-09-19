import os
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.optim import Adam
from tqdm import tqdm
import sys
import gc

# 确保能找到你的GNN模块
sys.path.append("..")
sys.path.append("../..")
from models.gnn import GNNGraph, graph_batch_from_smile
def parse_args():
    parser = argparse.ArgumentParser(description="Step-by-step Unified Contrastive Pre-training for ALL Embeddings")
    parser.add_argument('--dataset', default='mimic-iii', help='Dataset name')
    parser.add_argument('--dim', default=1536, type=int, help='Embedding dimension')
    parser.add_argument('--epochs', default=500, type=int, help='Epochs per entity type')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
    parser.add_argument('--output_dir', default='../pretrained_weights',
                        help='Output directory for all embeddings')
    parser.add_argument('--temp', default=0.1, type=float, help='Temperature for contrastive loss')
    return parser.parse_args()

def build_contextual_view(entity_type, voc_size, emb_dim, data_train):
    compute_device = torch.device('cpu')
    print(f"\nBuilding {entity_type.upper()} contextual view on CPU...")

    num_diag, num_proc, num_med = voc_size[0], voc_size[1], voc_size[2]

    # 根据目标实体类型，确定它的“语境提供者”
    if entity_type == 'diag':
        target_num, base_emb1, base_emb2 = num_diag, nn.Embedding(num_proc, emb_dim), nn.Embedding(num_med, emb_dim) #两个初始的embedding
        get_ctx_indices = lambda v: (v[1], v[2]) #如果是疾病，那么索引v 得到的是 手术 和药物的语境
        get_target_indices = lambda v: v[0]
    elif entity_type == 'proc':
        target_num, base_emb1, base_emb2 = num_proc, nn.Embedding(num_diag, emb_dim), nn.Embedding(num_med, emb_dim)
        get_ctx_indices = lambda v: (v[0], v[2])
        get_target_indices = lambda v: v[1]
    else:  # 'med'
        target_num, base_emb1, base_emb2 = num_med, nn.Embedding(num_diag, emb_dim), nn.Embedding(num_proc, emb_dim)
        get_ctx_indices = lambda v: (v[0], v[1])
        get_target_indices = lambda v: v[2]

    context_vectors = torch.zeros(target_num, emb_dim, device=compute_device)
    counts = torch.zeros(target_num, device=compute_device)

    for patient in tqdm(data_train, desc=f"Building {entity_type} context"):
        for visit in patient:
            target_indices = get_target_indices(visit)              #得到当前 疾病特征
            ctx_indices1, ctx_indices2 = get_ctx_indices(visit)     #

            if not target_indices or (not ctx_indices1 and not ctx_indices2): continue

            ctx_embs1 = base_emb1(torch.LongTensor(ctx_indices1))           #
            ctx_embs2 = base_emb2(torch.LongTensor(ctx_indices2))           #
            condition_vec = torch.cat([ctx_embs1, ctx_embs2]).mean(dim=0)   #

            for t_idx in target_indices:
                counts[t_idx] += 1
                context_vectors[t_idx] += (condition_vec - context_vectors[t_idx]) / counts[t_idx]

    print(f"{entity_type.upper()} contextual view built successfully.")
    return context_vectors.detach()

# def calculate_contrastive_loss(proj_A, proj_B, temperature=0.1):
#     # a: (N,d), b: (M,d)  —— 通常 N==M，若做词表级一一对齐则 N=M
#     a = F.normalize(a, dim=1)
#     b = F.normalize(b, dim=1)
#     logits = (a @ b.t()) / tau
#     # 一对一时，假定正样本配对是按索引对齐（0↔0, 1↔1, ...）
#     labels = torch.arange(min(a.size(0), b.size(0)), device=a.device)
#     loss_ab = F.cross_entropy(logits[:labels.numel()], labels)       # A→B
#     loss_ba = F.cross_entropy(logits.t()[:labels.numel()], labels)   # B→A
#     return 0.5 * (loss_ab + loss_ba)
def symmetric_contrastive_loss(z_base, z_ctx, tau=0.1):
    """
    z_base, z_ctx: (N, d) 已经 L2 normalize
    """
    logits = z_base @ z_ctx.T / tau
    labels = torch.arange(z_base.size(0), device=z_base.device)
    loss_base_to_ctx = F.cross_entropy(logits, labels)
    loss_ctx_to_base = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_base_to_ctx + loss_ctx_to_base)

def main():
    args = parse_args()
    output_dir = f"{args.output_dir}/{args.dataset}"
    print(output_dir)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 全局数据加载 ---
    print("Loading common data...")
    data_path = f'../../data/output/{args.dataset}/records_final.pkl'  #病人数据
    voc_path = f'../../data/output/{args.dataset}/voc_final.pkl'       #token大小
    molecule_path = f'../../data/input/{args.dataset}/atc3toSMILES.pkl'#分子SMILES结构

    with open(data_path, 'rb') as f:
        data = dill.load(f)
    data = [[visit[:3] for visit in patient] for patient in data]   #取病人前4次访问
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]

    with open(voc_path, 'rb') as f:
        voc = dill.load(f)
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = [len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word)]
    num_diag, num_proc, num_med = voc_size[0], voc_size[1], voc_size[2]

    with open(molecule_path, 'rb') as f:
        molecule_data = dill.load(f)
    smiles_list = [molecule_data[med_voc.idx2word[i]][0] for i in range(num_med) if
                   med_voc.idx2word[i] in molecule_data]

    # === 构建三类 contextual views ===
    diag_ctx = build_contextual_view('diag', voc_size, args.dim, data_train).to(device)
    proc_ctx = build_contextual_view('proc', voc_size, args.dim, data_train).to(device)
    med_ctx  = build_contextual_view('med',  voc_size, args.dim, data_train).to(device)

    # === Base Embeddings ===
    emb_diag = nn.Embedding(num_diag, args.dim).to(device)
    emb_proc = nn.Embedding(num_proc, args.dim).to(device)
    emb_med = nn.Embedding(num_med, args.dim).to(device)

    # === Projection heads (modality-specific) ===
    g_diag = nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU(), nn.Linear(args.dim, args.dim)).to(device)
    g_proc = nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU(), nn.Linear(args.dim, args.dim)).to(device)
    g_med = nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU(), nn.Linear(args.dim, args.dim)).to(device)

    optimizer = Adam(list(emb_diag.parameters()) +
                     list(emb_proc.parameters()) +
                     list(emb_med.parameters()) +
                     list(g_diag.parameters()) +
                     list(g_proc.parameters()) +
                     list(g_med.parameters()), lr=args.lr)

    # === 训练循环 ===
    for epoch in tqdm(range(args.epochs), desc="Unified Contrastive Pre-training"):
        # Base embeddings
        base_d = emb_diag.weight
        base_p = emb_proc.weight
        base_m = emb_med.weight

        # Contextual embeddings (固定)
        ctx_d = diag_ctx
        ctx_p = proc_ctx
        ctx_m = med_ctx

        # 投影 + normalize
        z_base_d = F.normalize(g_diag(base_d), dim=1)
        z_ctx_d = F.normalize(g_diag(ctx_d), dim=1)

        z_base_p = F.normalize(g_proc(base_p), dim=1)
        z_ctx_p = F.normalize(g_proc(ctx_p), dim=1)

        z_base_m = F.normalize(g_med(base_m), dim=1)
        z_ctx_m = F.normalize(g_med(ctx_m), dim=1)

        # 拼接 U
        z_base_all = torch.cat([z_base_d, z_base_p, z_base_m], dim=0)
        z_ctx_all = torch.cat([z_ctx_d, z_ctx_p, z_ctx_m], dim=0)

        loss = symmetric_contrastive_loss(z_base_all, z_ctx_all, tau=args.temp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # === 保存最终 embedding ===
    torch.save(F.normalize(g_diag(emb_diag.weight).detach().cpu(), dim=1),
               os.path.join(output_dir, 'pretrained_diag_embeddings.pt'))
    torch.save(F.normalize(g_proc(emb_proc.weight).detach().cpu(), dim=1),
               os.path.join(output_dir, 'pretrained_proc_embeddings.pt'))
    torch.save(F.normalize(g_med(emb_med.weight).detach().cpu(), dim=1),
               os.path.join(output_dir, 'pretrained_med_embeddings.pt'))

    print("\n\nAll pre-training phases completed successfully! Check the output directory for your embeddings.")

if __name__ == '__main__':
    main()


