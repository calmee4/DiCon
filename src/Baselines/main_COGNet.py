import os
import sys
import dill
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
sys.path.append("../..")
from utils.data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
from models.COGNet import COGNet
from utils.util import llprint, get_n_params, output_flatten, create_log_id, logging_config, get_model_path,ddi_rate_score_positive
from utils.util import sequence_output_process, multi_label_metric, ddi_rate_score
@torch.no_grad()
def eval_recommend_batch(model, batch_data, device, TOKENS, args):
    """
    用与训练一致的 encode/decode 流程做 *贪心* 生成，
    返回所有时间步的 logits（用于后续 output_flatten 解析）。
    """
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    (diseases, procedures, medications, visit_weights_patient, seq_length,
     d_length_matrix, p_length_matrix, m_length_matrix,
     d_mask_matrix, p_mask_matrix, m_mask_matrix,
     dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
     dec_proc, stay_proc, dec_proc_mask, stay_proc_mask) = batch_data

    # 按 vocab 进行 padding 替换并搬到 device
    diseases       = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
    procedures     = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
    dec_disease    = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
    stay_disease   = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
    dec_proc       = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
    stay_proc      = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)

    medications        = medications.to(device)
    m_mask_matrix      = m_mask_matrix.to(device)
    d_mask_matrix      = d_mask_matrix.to(device)
    p_mask_matrix      = p_mask_matrix.to(device)
    dec_disease_mask   = dec_disease_mask.to(device)
    stay_disease_mask  = stay_disease_mask.to(device)
    dec_proc_mask      = dec_proc_mask.to(device)
    stay_proc_mask     = stay_proc_mask.to(device)

    B = medications.size(0)
    V = medications.size(1)  # visit 数

    # 编码
    (input_disease_emb, input_proc_emb, encoded_med, cross_visit_scores,
     last_seq_med, last_m_mask, drug_memory) = model.encode(
        diseases, procedures, medications,
        d_mask_matrix, p_mask_matrix, m_mask_matrix,
        seq_length, dec_disease, stay_disease,
        dec_disease_mask, stay_disease_mask,
        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask,
        max_len=20
    )

    # 解码（贪心）
    partial_input = torch.full((B, V, 1), SOS_TOKEN, device=device)
    partial_logits = None

    for _ in range(args.max_len):
        L = partial_input.size(2)
        partial_m_mask = torch.zeros((B, V, L), device=device).float()

        partial_logits = model.decode(
            partial_input, input_disease_emb, input_proc_emb, encoded_med,
            last_seq_med, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask, last_m_mask,
            drug_memory
        )  # [B, V, L, n_meds(+special)]

        # 取当前步的 token（贪心）
        _, nxt = torch.topk(partial_logits[:, :, -1, :], 1, dim=-1)  # [B,V,1]
        partial_input = torch.cat([partial_input, nxt], dim=-1)

    return partial_logits


@torch.no_grad()
def eval(args, epoch, model, eval_dataloader, voc_size, ddi_adj_path, rec_results_path=None):
    """
    与训练/验证一致的评估：
      返回: (ddi_rate, ja, prauc, avg_f1, avg_med, ddi_rate_positive)
    """
    device = torch.device(f'cuda:{args.cuda}')
    END_TOKEN      = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN  = voc_size[2] + 2
    SOS_TOKEN      = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    rec_results = []  # 仅 test 模式落盘

    # NaN 安全的平均函数：空/NaN -> 0
    def safe_mean_probs(pred2d, K):
        if pred2d is None:
            return np.zeros(K, dtype=np.float32)
        arr = np.array(pred2d, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return np.zeros(K, dtype=np.float32)
        # 只取真实药物部分（去掉 SOS/END 两个特殊位）
        arr = arr[:, :K]
        m = np.nanmean(arr, axis=0)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    for idx, batch in enumerate(tqdm(eval_dataloader, ncols=60, desc="Evaluation")):
        # DataLoader 给的是带 visit 权重的 batch
        (diseases, procedures, medications, visit_weights_patient, seq_length,
         d_length_matrix, p_length_matrix, m_length_matrix,
         d_mask_matrix, p_mask_matrix, m_mask_matrix,
         dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
         dec_proc, stay_proc, dec_proc_mask, stay_proc_mask) = batch

        visit_cnt += seq_length.sum().item()

        # 贪心生成整段 logits
        logits = eval_recommend_batch(model, batch, device, TOKENS, args)

        # 解析为每次就诊的标签与“每步的分布”
        # 注意：这里 voc_out 传 voc_size[2]（真实药物数），END_TOKEN 指结束位
        labels, predictions = output_flatten(
            medications, logits, seq_length, m_length_matrix,
            voc_size[2], END_TOKEN, device,
            training=False, testing=False, max_len=args.max_len
        )

        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        for label, pred_steps in zip(labels, predictions):
            # label 是真实集合
            gt_vec = np.zeros(voc_size[2], dtype=np.int64)
            gt_vec[label] = 1
            y_gt.append(gt_vec)

            # pred_steps: [T, vocab(+special)] 的每步分布（logits 或 log-prob）
            # 用 util 里的工具把生成序列转集合
            out_list, sorted_predict = sequence_output_process(
                pred_steps, [voc_size[2], voc_size[2] + 1]
            )

            # 概率向量（对每个药，按时间步做平均；若空则全零）
            prob_vec = safe_mean_probs(pred_steps, voc_size[2])
            y_pred_prob.append(prob_vec)

            # 0/1 预测向量（集合化）
            pred_vec = np.zeros(voc_size[2], dtype=np.int64)
            pred_vec[out_list] = 1
            y_pred.append(pred_vec)

            y_pred_label.append(sorted(sorted_predict))
            med_cnt += len(sorted_predict)

        # 记录给 DDI 用
        smm_record.append(y_pred_label)

        # 指标
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja); prauc.append(adm_prauc)
        avg_p.append(adm_avg_p); avg_r.append(adm_avg_r); avg_f1.append(adm_avg_f1)

        # 权重（每个病人的最大就诊权重）
        visit_weights.append(torch.max(visit_weights_patient, dim=1)[0].item())

        # 如需落盘（仅测试时）
        if args.test and rec_results_path:
            rec_results.append([
                diseases[0].tolist(), procedures[0].tolist(), medications[0].tolist(),
                y_pred_label, visit_weights_patient.tolist(), [adm_ja]
            ])

        llprint(f'\rtest step: {idx + 1} / {len(eval_dataloader)}')

    # 写 rec 结果（可选）
    if args.test and rec_results_path:
        os.makedirs(rec_results_path, exist_ok=True)
        dill.dump(rec_results, open(os.path.join(rec_results_path, 'rec_results.pkl'), 'wb'))

    # DDI 指标
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)
    ddi_rate_positive = ddi_rate_score_positive(
        smm_record, path=f"../../data/output/{args.dataset}/ddi_B_final.pkl"
    )

    # 汇总日志
    logging.info(
        f"Epoch {epoch:03d}, "
        f"Jaccard: {np.mean(ja):.4f}, "
        f"DDI Rate: {ddi_rate:.4f}, "
        f"DDI Rate Positive: {ddi_rate_positive:.4f}, "
        f"PRAUC: {np.mean(prauc):.4f}, "
        f"AVG_PRC: {np.mean(avg_p):.4f}, "
        f"AVG_RECALL: {np.mean(avg_r):.4f}, "
        f"AVG_F1: {np.mean(avg_f1):.4f}, "
        f"AVG_MED: {med_cnt / visit_cnt:.4f}"
    )

    # 返回顺序与你 main() 里使用的一致
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt, ddi_rate_positive

# Training settings
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default=' ', help="User notes")
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-s', '--single', action='store_true', default=False, help="single visit")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log0", help='log dir prefix like "log0"')
    parser.add_argument('--model_name', type=str, default="COGNet", help="model name")
    parser.add_argument('--dataset', type=str, default="mimic-iii", help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')
    parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop after this many epochs without improvement')
    parser.add_argument('--cuda', type=int, default=2 , help='which cuda')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of node embedding(randomly initialize)')

    args = parser.parse_args()
    return args
@torch.no_grad()
def test_bootstrap(
    args, model, device, data_test_raw, voc_size,
    *, ddi_adj_path, rounds=10, ratio=0.8, seed=0
):
    """
    对 test 集做 bootstrap：每轮抽取 ratio*N 个病人（有放回），共 rounds 次。
    指标来源：本文件的 eval（与训练/验证完全一致）。
    返回：无（打印并写日志），也可以按需返回 mean/std。
    """
    import numpy as np, time
    from torch.utils.data import DataLoader
    from utils.data_loader import mimic_data, pad_batch_v2_eval
    from utils.util import get_n_params

    model = model.to(device).eval()
    print('--------------------Begin Testing (bootstrap)--------------------')

    rng = np.random.default_rng(seed)
    N = len(data_test_raw)
    k = int(round(N * ratio))
    print("calculate ratio ")
    ds = mimic_data(data_test_raw)
    dl = DataLoader(ds, batch_size=1, collate_fn=pad_batch_v2_eval,
                    shuffle=False, pin_memory=True)
    ddi_rate, ja, prauc, avg_f1, avg_med, ddi_rate_pos = eval(
        args, 0, model, dl, voc_size, ddi_adj_path, rec_results_path=None
    )
    print("calcualte over")
    print("calculate over")
    rows = []
    tic = time.time()
    for r in range(rounds):
        idx = rng.choice(N, size=k, replace=True).tolist()
        subset = [data_test_raw[i] for i in idx]

        ds = mimic_data(subset)
        dl = DataLoader(ds, batch_size=1, collate_fn=pad_batch_v2_eval,
                        shuffle=False, pin_memory=True)

        # 直接用上面定义的 eval（不落盘）
        ddi_rate, ja, prauc, avg_f1, avg_med, ddi_rate_pos = eval(
            args, 0, model, dl, voc_size, ddi_adj_path, rec_results_path=None
        )
        rows.append([ja, ddi_rate, ddi_rate_pos, avg_f1, prauc, avg_med])
        llprint(f'\rbootstrap round: {r+1}/{rounds}')

    arr = np.array(rows, dtype=np.float64)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    names = ['ja', 'ddi_rate', 'ddi_rate_positive', 'avg_f1', 'prauc', 'med']
    out = '\n'.join(f'{n}:\t{m:.4f} ± {s:.4f}' for n, (m, s) in zip(names, zip(mean, std)))

    print('\n' + out); logging.info(out)
    print('avg time/round:', (time.time() - tic) / rounds)
    print('parameters', get_n_params(model))


def main(args):
    # set logger
    if args.test:
        args.note = f'test of {args.log_dir_prefix}'
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # load data
    data_path = f'../../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}' + '/ddi_A_final.pkl'
    ehr_adj_path = f'../../data/output/{args.dataset}' + '/ehr_adj_final.pkl'
    ddi_mask_path = f'../../data/output/{args.dataset}' + '/ddi_mask_H.pkl'

    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1
    
    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x:med_count[x])
            data[i][j][2] = cur_medications


    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    val_len = int(len(data[split_point:]) / 2)
    data_val = data[split_point:split_point + val_len]
    data_test = data[split_point+val_len:]
    if args.single:
        data_train = [[visit] for patient in data_train for visit in patient]
        data_val = [[visit] for patient in data_val for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_val)
    test_dataset = mimic_data(data_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train, shuffle=False, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False, pin_memory=True)
    
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    # model initialization
    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    model = COGNet(args, voc_size, ehr_adj, ddi_adj, ddi_mask_H)
    logging.info(model)
         
    # test
    if args.test:
        # model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model_path = './saved/mimic-iii/COGNet/Epoch_43_JA_0.5233_DDI_0.07859.model'
        model.load_state_dict(torch.load(open(model_path, 'rb')))
        model.to(device=device)

        eval(args, 0, model, test_dataloader, voc_size, ddi_adj_path, save_dir+'/rec_results.pkl')
        return
    else:
        writer = SummaryWriter(save_dir) # 自动生成log文件夹

    # train and validation
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    print('parameters', get_n_params(model))
    # tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch=0)
    # model.load_state_dict
    # Test(model, device, data_test, voc_size, data)

    EPOCH = 50
    best_epoch, best_ja = 0, 0
    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, lr={args.lr}, '
            f'batch_size={args.batch_size}, beam_size={args.beam_size}, max_med_len={args.max_len}, logger={log_save_id}')
        model.train()
        tic = time.time()
        loss_train, loss_val = 0, 0
        for idx, data in enumerate(train_dataloader):

            diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data

            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)
            output_logits = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)

            loss = F.nll_loss(predictions, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()/len(train_dataloader)
            llprint('\rtraining step: {} / {}'.format(idx, len(train_dataloader)))

        with torch.no_grad():
            for idx, data in tqdm(enumerate(eval_dataloader), ncols=60, desc="Val loss", total=len(eval_dataloader)):  # every patient
                diseases, procedures, medications, visit_weights_patient, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)
            output_logits = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
            loss = F.nll_loss(predictions, labels.long())
            loss_val += loss.item()/len(eval_dataloader)

        # logging.info(f'loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}')
        # evaluation

        tic2 = time.time()
        ddi_rate, ja, prauc, avg_f1, avg_med,ddi_rate_positive  = eval(args, epoch, model, eval_dataloader, voc_size, ddi_adj_path)
        logging.info('training time: {:.1f}, test time: {:.1f}'.format(time.time() - tic, time.time() - tic2))
        # print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))
        tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                          loss_train, loss_val,ddi_rate_positive)
        # ddi_rate_positive = ddi_rate_score_positive(smm_record, path="../../data/output/mimic-iii/ddi_B_final.pkl")
        # save best epoch
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_prauc, best_ddi_rate, best_avg_med = ja, prauc, ddi_rate, avg_med
            best_model_state = deepcopy(model.state_dict()) 
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))
        # print ('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))

        if epoch - best_epoch > args.early_stop:   # n个epoch内，验证集性能不上升之后就停
            break

    # save best model
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, ddi_rate)), 'wb'))
    model.load_state_dict(best_model_state)
    model.to(device=device)
    # eval(args, 0, model, test_dataloader, voc_size, ddi_adj_path, save_dir + '/rec_results.pkl')
    test_bootstrap(
        args, model, device, data_test, voc_size,
        ddi_adj_path=ddi_adj_path, rounds=10, ratio=0.8, seed=0
    )

def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                        loss_train=0, loss_val=0,ddi_rate_positve=0):
    if epoch > 0:
        writer.add_scalar('Loss/Train', loss_train, epoch)
        writer.add_scalar('Loss/Val', loss_val, epoch)

    writer.add_scalar('Metrics/Jaccard', ja, epoch)
    writer.add_scalar('Metrics/prauc', prauc, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate, epoch)
    writer.add_scalar('Metrics/Med_count', avg_med, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate_positve, epoch)
if __name__ == '__main__':

    torch.manual_seed()
    np.random.seed()
    args = get_args()
    main(args)
